"""Typed application settings, loaded from environment / .env.

Every knob that changes behaviour (refusal threshold, retrieval k, provider order)
lives here so it is visible, documented, and reproducible in an eval run — not
scattered as magic numbers across the codebase.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from verity.types import Provider


class ModelPrice(BaseModel, frozen=True):
    """List price of one model, in USD per million tokens."""

    input_usd_per_mtok: float = Field(ge=0.0)
    output_usd_per_mtok: float = Field(ge=0.0)


# Anthropic list prices from the Anthropic model table; OpenAI's from its gpt-4.1-mini
# model page. Override with VERITY_MODEL_PRICES='{"<model>": {...}}' when a contract
# or a model change makes them wrong: cost figures are only as good as this table.
DEFAULT_MODEL_PRICES: dict[str, ModelPrice] = {
    "claude-sonnet-4-6": ModelPrice(input_usd_per_mtok=3.0, output_usd_per_mtok=15.0),
    "claude-opus-5-5": ModelPrice(input_usd_per_mtok=4.0, output_usd_per_mtok=20.0),
    "gpt-4.1-mini": ModelPrice(input_usd_per_mtok=0.40, output_usd_per_mtok=1.60),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VERITY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Database ---------------------------------------------------------------------
    # Holds the password: kept out of repr, dumps and logs.
    database_url: SecretStr = SecretStr("postgresql://verity:verity@localhost:5432/verity")
    # pgvector is what the API, the demo and the reference eval run on (ADR 0002).
    # "memory" runs without a database: unit tests, the CI pipeline smoke test.
    store: Literal["pgvector", "memory"] = "pgvector"

    # --- Embeddings (local by default → CI needs no API keys) -------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # --- Reranking (local cross-encoder, offline) -------------------------------------
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Retrieval --------------------------------------------------------------------
    retrieval_k: int = 20  # candidates per retriever before fusion
    rerank_k: int = 8  # kept after reranking → fed to the agent
    rrf_k: int = Field(
        default=60,
        gt=0,
        description="Reciprocal rank fusion constant; ranks are 0-based (see retrieval.fusion).",
    )

    # --- LLM runtime ------------------------------------------------------------------
    provider_order: tuple[Provider, ...] = (Provider.ANTHROPIC, Provider.OPENAI)
    # SecretStr keeps keys out of repr()/str()/model_dump() output, so a logged or
    # printed Settings object cannot leak them. Unwrap only at the SDK boundary.
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    anthropic_model: str = "claude-sonnet-4-6"
    openai_model: str = "gpt-4.1-mini"
    # LLM-as-judge (eval --judge live). "auto": the other family (OpenAI) when its key is
    # set, else Claude. One client, no fallback; a judge equal to the agent's model is
    # refused at start (verity.llm.routing.create_judge_client).
    judge_provider: Literal["auto", "anthropic", "openai"] = "auto"
    judge_anthropic_model: str = "claude-opus-5-5"
    judge_openai_model: str = "gpt-4.1-mini"
    # A hung provider must surface as ProviderUnavailableError in bounded time, so the
    # router can fall back; the SDK default (10 min) would hold a request that long.
    # Worst case per provider: llm_timeout_s * (llm_max_retries + 1).
    llm_timeout_s: float = Field(default=30.0, gt=0.0)
    llm_max_retries: int = Field(default=2, ge=0)
    model_prices: dict[str, ModelPrice] = Field(default_factory=lambda: dict(DEFAULT_MODEL_PRICES))
    # Default cap for one `verity ask`. A live answer on the default model cost
    # $0.012 on average in the run of record (scorecards/live-v1.0.0.json); $1 stops a
    # runaway answer without getting in the way of a normal one.
    ask_budget_usd: float = Field(default=1.0, gt=0.0)

    # --- Refusal (headline feature) ---------------------------------------------------
    # The dev rule fixed before the v1.1.0 runs (highest refusal recall at precision
    # >= 0.9, else highest F1) chose 0.0; at 0.0 no score is below it, so refusals come
    # from the deterministic checks and the scorer's own refusal only.
    confidence_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Below this scorer-reported confidence the agent refuses to answer. The "
            "score lies in [0, 1], so the default 0 refuses nothing."
        ),
    )
    max_agent_steps: int = 4  # cap on multi-step retrieval decomposition
    # Deterministic checks a cited draft must pass before the scorer runs
    # (verity.agent.support). On the committed live run (d0ca1d5), the 12 answered
    # questions had coverage 0.58-1.00 and a best cited reranker logit of 0.94-9.31;
    # q-015, labelled out of scope and answered, had 0.30 and -0.85. Both values were
    # chosen on those same 16 examples: they need a held-out split to be validated.
    min_citation_coverage: float = Field(default=0.5, ge=0.0, le=1.0)
    min_cited_rerank_score: float = 0.0
    # Off by default: as the first failing check on the dev runs it refused 11
    # answerable records (4 distinct questions) for 5 unanswerable ones (2), and
    # replayed alone on the naive arm's dev drafts 3 answerable for 0 unanswerable
    # (scripts/gate_ablation.py).
    irrelevant_citation_check: bool = False

    # --- HTTP API (see docs/adr/0006-api-security-baseline.md) ------------------------
    # The longest labeled question is 97 characters. 2000 leaves 20x headroom while
    # bounding what one request puts into each of its three prompts (~500 tokens).
    api_max_question_chars: int = Field(default=2000, gt=0)
    # The shipped corpus files are 1-4 KB; 1 MiB fits a long contract or a short PDF.
    api_max_upload_bytes: int = Field(default=1_048_576, gt=0)
    # The corpus is shared by every client and, on pgvector, outlives the process; this
    # caps its growth because no endpoint deletes a document.
    api_max_documents: int = Field(default=200, gt=0)
    # JSON list in the environment: VERITY_API_KEYS='["key-1","key-2"]'.
    api_keys: tuple[SecretStr, ...] = ()
    # No key configured is a startup error unless this is set: an open API must be a
    # deliberate choice (local development), never the result of a missing variable.
    api_auth_disabled: bool = False
    api_rate_limit_per_minute: int = Field(default=30, gt=0)
    # Wrong or missing keys per client address per minute: enough for a mistyped key,
    # too few to guess one.
    api_auth_failures_per_minute: int = Field(default=10, gt=0)
    # Spending budget per key over a sliding window, from the cost each /ask reports.
    # $5 an hour is about 400 live answers at the run-of-record average ($0.012): room
    # for a team's interactive use, a ceiling on a leaked or looping key.
    api_budget_usd: float = Field(default=5.0, gt=0.0)
    api_budget_window_s: float = Field(default=3600.0, gt=0.0)

    # --- Observability ----------------------------------------------------------------
    # None → spans stay in the in-process exporter only (backs /metrics).
    otel_endpoint: str | None = None
    # Spans kept in memory for /metrics and /dashboard; older spans are dropped.
    trace_max_spans: int = Field(default=10_000, gt=0)
    log_level: str = "INFO"

    def price_for(self, model: str) -> ModelPrice:
        """The configured price of ``model``; an unpriced model is a configuration error."""
        try:
            return self.model_prices[model]
        except KeyError:
            raise ValueError(
                f"no price configured for model {model!r}; add it to VERITY_MODEL_PRICES"
            ) from None

    @property
    def has_llm_api_key(self) -> bool:
        """True when at least one provider key is set (see :func:`key_is_set`)."""
        return any(key_is_set(key) for key in (self.anthropic_api_key, self.openai_api_key))


def key_is_set(key: SecretStr | None) -> bool:
    """An empty ``VERITY_*_API_KEY=`` line (as in ``.env.example``) counts as unset."""
    return key is not None and bool(key.get_secret_value())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
