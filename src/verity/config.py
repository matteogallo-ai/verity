"""Typed application settings, loaded from environment / .env.

Every knob that changes behaviour (refusal threshold, retrieval k, provider order)
lives here so it is visible, documented, and reproducible in an eval run — not
scattered as magic numbers across the codebase.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from verity.types import Provider


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VERITY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Database ---------------------------------------------------------------------
    database_url: str = "postgresql://verity:verity@localhost:5432/verity"

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
    judge_model: str = "claude-sonnet-4-6"  # LLM-as-judge for faithfulness eval

    # --- Refusal (headline feature) ---------------------------------------------------
    confidence_threshold: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="Below this scorer-reported confidence the agent refuses to answer.",
    )
    max_agent_steps: int = 4  # cap on multi-step retrieval decomposition

    # --- Observability ----------------------------------------------------------------
    otel_endpoint: str | None = None  # None → console exporter
    log_level: str = "INFO"

    @property
    def has_llm_api_key(self) -> bool:
        """True when at least one provider key is set to a non-empty value.

        An empty ``VERITY_*_API_KEY=`` line (as in ``.env.example``) counts as unset.
        """
        return any(
            key is not None and key.get_secret_value()
            for key in (self.anthropic_api_key, self.openai_api_key)
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
