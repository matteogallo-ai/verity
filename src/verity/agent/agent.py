"""Top-level agentic RAG orchestrator.

Pipeline for one ``answer(question)`` call:

1. **Decompose** the question into ordered sub-questions (single element if simple).
2. **Retrieve** per sub-question through the hybrid retriever; deduplicate hits by
   chunk id while preserving the highest-scoring occurrence.
3. **Synthesize** a draft answer + validated citations grounded in the aggregate
   evidence.
4. **Score** confidence and refuse below-threshold rather than shipping a
   fabricated answer.

Guarantees on the returned :class:`Answer`:
- non-refusal answers carry ≥1 citation whose ``quote`` is a verbatim substring of
  the referenced chunk (enforced upstream in :class:`CitedSynthesizer`);
- ``confidence.refused = True`` iff calibrated confidence < threshold — the text
  becomes an honest "I don't know" with a rationale;
- ``hits_used`` and ``usage`` are always populated for observability and eval.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import structlog

from verity.agent.base import Agent, ConfidenceScorer, QuestionDecomposer
from verity.agent.decomposer import LLMQuestionDecomposer
from verity.agent.scorer import LLMConfidenceScorer
from verity.agent.synthesizer import CitedSynthesizer
from verity.config import get_settings
from verity.llm.base import RoutingClient
from verity.retrieval.base import Retriever
from verity.types import Answer, RetrievalHit, UsageStats, new_id

log = structlog.get_logger(__name__)

_REFUSAL_TEXT = "I don't know based on the provided documents."


@dataclass(frozen=True)
class AgentDeps:
    """The four components the agent composes. Injectable — the CLI wires the real
    ones; tests substitute a stub LLM + in-memory retriever."""

    retriever: Retriever
    decomposer: QuestionDecomposer
    synthesizer: CitedSynthesizer
    scorer: ConfidenceScorer


class RagAgent(Agent):
    """The public entry point the CLI (and future API) calls."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        decomposer: QuestionDecomposer,
        synthesizer: CitedSynthesizer,
        scorer: ConfidenceScorer,
        rerank_k: int | None = None,
    ) -> None:
        settings = get_settings()
        self._retriever = retriever
        self._decomposer = decomposer
        self._synthesizer = synthesizer
        self._scorer = scorer
        self._rerank_k = rerank_k if rerank_k is not None else settings.rerank_k

    async def answer(self, question: str) -> Answer:
        started = time.perf_counter()
        trace_id = new_id()
        sub_questions = await self._decomposer.decompose(question)
        log.info("decomposed", trace_id=str(trace_id), n_sub_questions=len(sub_questions))

        hits = await self._collect_hits(sub_questions)
        log.info("aggregated_hits", trace_id=str(trace_id), n_hits=len(hits))

        synthesis = await self._synthesizer.synthesize(question, hits)
        confidence = await self._scorer.score(question, synthesis.text, hits)

        # Aggregate token/cost accounting across every LLM call in the flow:
        # decomposer + synthesizer + scorer (one call each in the current
        # single-sub-question pipeline). Latency is the measured wall-clock,
        # not a sum of per-call latencies — the individual calls run partially
        # overlapped, so summing would over-count.
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        usage = _aggregate_usage(
            _last_usage_of(self._decomposer),
            synthesis.usage,
            _last_usage_of(self._scorer),
            latency_ms=latency_ms,
        )

        if confidence.refused:
            log.info(
                "refused",
                trace_id=str(trace_id),
                score=confidence.score,
                rationale=confidence.rationale,
            )
            return Answer(
                query=question,
                text=_REFUSAL_TEXT,
                citations=(),
                confidence=confidence,
                hits_used=tuple(hits),
                usage=usage,
                trace_id=trace_id,
            )

        return Answer(
            query=question,
            text=synthesis.text or _REFUSAL_TEXT,
            citations=synthesis.citations,
            confidence=confidence,
            hits_used=tuple(hits),
            usage=usage,
            trace_id=trace_id,
        )

    async def _collect_hits(self, sub_questions: list[str]) -> list[RetrievalHit]:
        """Retrieve per sub-question and deduplicate by chunk id, keeping the best score.

        We iterate serially rather than gather() so log lines interleave predictably
        with the per-stage HybridRetriever spans — helpful for demo transcripts. This
        can move to gather() once observability lands (S5) and we don't rely on log
        ordering for the demo.
        """
        seen: dict[str, RetrievalHit] = {}
        for sub in sub_questions:
            hits = await self._retriever.retrieve(sub, k=self._rerank_k)
            for hit in hits:
                cid = str(hit.chunk.id)
                current = seen.get(cid)
                if current is None or hit.score > current.score:
                    seen[cid] = hit
        # Preserve the best-score-first ordering so the synthesizer sees the strongest
        # evidence at the top of its rendered evidence block.
        return sorted(seen.values(), key=lambda h: -h.score)


def _last_usage_of(component: object) -> UsageStats | None:
    """Best-effort accessor for the ``last_usage`` attribute the concrete LLM-backed
    decomposer / scorer expose. Protocol implementations without it contribute nothing
    to the aggregate — which is the honest fallback (better than fabricating a count)."""
    usage = getattr(component, "last_usage", None)
    return usage if isinstance(usage, UsageStats) else None


def _aggregate_usage(*parts: UsageStats | None, latency_ms: float) -> UsageStats:
    """Sum every non-null ``UsageStats`` into one aggregate — every LLM call counted."""
    input_tokens = 0
    output_tokens = 0
    llm_calls = 0
    cost_usd = 0.0
    for part in parts:
        if part is None:
            continue
        input_tokens += part.input_tokens
        output_tokens += part.output_tokens
        llm_calls += part.llm_calls
        cost_usd += part.cost_usd
    return UsageStats(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        llm_calls=llm_calls,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
    )


def create_default_agent(
    *,
    retriever: Retriever,
    client: RoutingClient,
) -> RagAgent:
    """Wire the default agent: LLM-based decomposer + synthesizer + scorer."""
    return RagAgent(
        retriever=retriever,
        decomposer=LLMQuestionDecomposer(client),
        synthesizer=CitedSynthesizer(client),
        scorer=LLMConfidenceScorer(client),
    )


__all__ = ["AgentDeps", "RagAgent", "create_default_agent"]
