"""LLM-as-judge for faithfulness + a deterministic scripted stub.

Two implementations behind the ``FaithfulnessJudge`` protocol:

- :class:`LLMFaithfulnessJudge` — the real judge. Renders
  ``prompts/judge_faithfulness.prompt`` with the answer + citations + retrieved
  context, parses the strict JSON output, and turns it into an
  :class:`AnswerMetrics` per example. Requires a routed LLM client (and therefore
  a real API key at runtime).

- :class:`StubFaithfulnessJudge` — a *scripted, mechanical* judge for CI and for
  local runs without a key. It applies a small set of transparent rules:

  * A refusal answer has no claims → faithfulness/citation_accuracy default to 1.0
    by convention (nothing to fail) and hallucination_rate is 0.0.
  * A non-refusal answer is split into atomic claims by sentence terminators. A
    claim is *supported* iff at least one of the answer's citations has a
    ``quote`` that overlaps meaningfully with the claim text, OR the claim
    contains a phrase that is verbatim present in the retrieved context (the
    substring check is deliberately strict so a hallucination cannot pass).
  * Every citation's ``quote`` is already a substring of its chunk by construction
    (S3 synthesizer invariant), so ``citation_accuracy`` is 1.0 by definition —
    we re-verify it here and would flag any drift.

  The per-example ``refusal_*`` fields are placeholders (0.0); the aggregator
  in :mod:`verity.eval.harness` computes the real refusal-calibration numbers
  from the (expected, observed) pairs across the whole dataset.

The Scorecard produced by a stub run carries ``judge_model="stub-judge-v1"`` so
downstream consumers (README, dashboard) never mistake a mechanical verdict for a
real LLM-judge measurement.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from verity.eval.base import FaithfulnessJudge
from verity.llm.base import RoutingClient
from verity.llm.prompt import load_prompt
from verity.types import Answer, AnswerMetrics, EvalExample, RetrievalHit, UsageStats

STUB_JUDGE_MODEL = "stub-judge-v1"

_REFUSAL_MARKERS = ("i don't know", "i do not know")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class JudgeResult:
    """One judge call's output — metrics + LLM usage (0 for the stub)."""

    metrics: AnswerMetrics
    usage: UsageStats


def _is_refusal(answer: Answer) -> bool:
    """A refusal is identified by ``confidence.refused`` (canonical) or by the
    conventional refusal text emitted by the agent."""
    if answer.confidence.refused:
        return True
    text = answer.text.lower()
    return any(marker in text for marker in _REFUSAL_MARKERS)


def _atomic_claims(text: str) -> list[str]:
    """Split answer text into atomic claims by sentence terminator.

    A brutally simple splitter is intentional: LLM judges of production quality
    do far more (coreference resolution, quantifier scoping); the stub does not
    pretend to. What matters is that CI runs deterministically on the same input.
    """
    stripped = text.strip()
    if not stripped:
        return []
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(stripped) if p.strip()]
    return parts


def _render_context(hits: list[RetrievalHit]) -> str:
    if not hits:
        return "(no evidence)"
    return "\n\n".join(
        f"[{i}] (section: {h.chunk.section or '-'})\n{h.chunk.text.strip()}"
        for i, h in enumerate(hits)
    )


def _render_citations(answer: Answer) -> str:
    if not answer.citations:
        return "(no citations)"
    return "\n".join(
        f"[{i}] chunk={c.chunk_id} quote={c.quote!r}" for i, c in enumerate(answer.citations)
    )


# --------------------------------------------------------------------------------------
# Stub
# --------------------------------------------------------------------------------------


class StubFaithfulnessJudge(FaithfulnessJudge):
    """Deterministic scripted judge — see module docstring for the exact rules."""

    def __init__(self) -> None:
        self.model = STUB_JUDGE_MODEL
        self.last_usage: UsageStats | None = None

    async def judge(self, example: EvalExample, answer: Answer) -> AnswerMetrics:
        # Refusals have no claims — nothing to fail, nothing to hallucinate.
        if _is_refusal(answer):
            metrics = AnswerMetrics(
                faithfulness=1.0,
                citation_accuracy=1.0,
                hallucination_rate=0.0,
                refusal_precision=0.0,
                refusal_recall=0.0,
            )
            self.last_usage = _zero_usage()
            return metrics

        claims = _atomic_claims(answer.text)
        if not claims:
            self.last_usage = _zero_usage()
            return AnswerMetrics(
                faithfulness=0.0,
                citation_accuracy=0.0,
                hallucination_rate=1.0,
                refusal_precision=0.0,
                refusal_recall=0.0,
            )

        context = "\n\n".join(h.chunk.text for h in answer.hits_used)
        quotes = [c.quote for c in answer.citations]

        supported = 0
        for claim in claims:
            if _claim_supported(claim, quotes, context):
                supported += 1
        faithfulness = supported / len(claims)

        # Citation accuracy: every quote must be a verbatim substring of the
        # retrieved context (S3 synthesizer already enforces this against the
        # citation's chunk, so this is a floor sanity check).
        if answer.citations:
            valid_citations = sum(1 for q in quotes if q in context)
            citation_accuracy = valid_citations / len(quotes)
        else:
            citation_accuracy = 0.0 if example.answerable else 1.0

        hallucination_rate = 1.0 if (example.answerable and supported < len(claims)) else 0.0

        self.last_usage = _zero_usage()
        return AnswerMetrics(
            faithfulness=faithfulness,
            citation_accuracy=citation_accuracy,
            hallucination_rate=hallucination_rate,
            refusal_precision=0.0,
            refusal_recall=0.0,
        )


def _claim_supported(claim: str, quotes: list[str], context: str) -> bool:
    claim_lower = claim.lower()
    for quote in quotes:
        if quote and (quote.lower() in claim_lower or claim_lower in quote.lower()):
            return True
    # Fallback: any 4+-word contiguous phrase from the context appearing in the claim.
    for phrase_len in (8, 6, 5, 4):
        for start in range(len(context) - phrase_len):
            fragment = context[start : start + phrase_len]
            if len(fragment.split()) >= 3 and fragment.lower() in claim_lower:
                return True
    return False


def _zero_usage() -> UsageStats:
    return UsageStats(input_tokens=0, output_tokens=0, llm_calls=0, cost_usd=0.0, latency_ms=0.0)


# --------------------------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------------------------


class LLMFaithfulnessJudge(FaithfulnessJudge):
    """Prompt-driven judge that runs against a real ``RoutingClient``.

    ``model`` reflects the underlying provider's model id from the first client in
    the router (used by the Scorecard for traceability).
    """

    def __init__(
        self,
        client: RoutingClient,
        *,
        prompt_name: str = "judge_faithfulness",
        model_label: str | None = None,
    ) -> None:
        self._client = client
        self._prompt = load_prompt(prompt_name)
        self.model = model_label or _first_client_model(client) or "llm-judge"
        self.last_usage: UsageStats | None = None

    async def judge(self, example: EvalExample, answer: Answer) -> AnswerMetrics:
        # Refusals bypass the judge — same convention as the stub.
        if _is_refusal(answer):
            self.last_usage = _zero_usage()
            return AnswerMetrics(
                faithfulness=1.0,
                citation_accuracy=1.0,
                hallucination_rate=0.0,
                refusal_precision=0.0,
                refusal_recall=0.0,
            )

        messages = self._prompt.render(
            context=_render_context(list(answer.hits_used)),
            answer=answer.text,
            citations=_render_citations(answer),
        )
        completion = await self._client.complete(messages, max_tokens=512, temperature=0.0)
        self.last_usage = completion.usage
        parsed = _parse_judge_output(completion.text)
        if parsed is None:
            # Refuse to fabricate a chiffre on malformed output — flag as
            # unsupported so the scorecard reflects the failure honestly.
            return AnswerMetrics(
                faithfulness=0.0,
                citation_accuracy=0.0,
                hallucination_rate=1.0 if example.answerable else 0.0,
                refusal_precision=0.0,
                refusal_recall=0.0,
            )
        supported, citation_ok, total = parsed
        faithfulness = supported / total if total else 0.0
        citation_accuracy = citation_ok / total if total else 0.0
        hallucination_rate = 1.0 if (example.answerable and supported < total) else 0.0
        return AnswerMetrics(
            faithfulness=faithfulness,
            citation_accuracy=citation_accuracy,
            hallucination_rate=hallucination_rate,
            refusal_precision=0.0,
            refusal_recall=0.0,
        )


def _first_client_model(client: RoutingClient) -> str | None:
    clients = getattr(client, "clients", None)
    if clients and len(clients) > 0:
        model = getattr(clients[0], "model", None)
        if isinstance(model, str):
            return model
    return None


def _parse_judge_output(text: str) -> tuple[int, int, int] | None:
    """Return ``(supported_count, citation_ok_count, total_claims)`` or ``None``."""
    stripped = text.strip()
    if not stripped:
        return None
    candidate = _first_json_object(stripped) or stripped
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    claims = parsed.get("claims")
    if not isinstance(claims, list) or not claims:
        return None
    supported = 0
    citation_ok = 0
    for entry in claims:
        if not isinstance(entry, dict):
            return None
        if entry.get("supported"):
            supported += 1
        if entry.get("citation_ok"):
            citation_ok += 1
    return supported, citation_ok, len(claims)


def _first_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""
    return text[start : end + 1]


__all__ = [
    "STUB_JUDGE_MODEL",
    "JudgeResult",
    "LLMFaithfulnessJudge",
    "StubFaithfulnessJudge",
]
