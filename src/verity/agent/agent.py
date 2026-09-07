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
from verity.observability.base import Tracer
from verity.observability.logging import request_trace
from verity.observability.tracer import NoOpTracer
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
        tracer: Tracer | None = None,
    ) -> None:
        settings = get_settings()
        self._retriever = retriever
        self._decomposer = decomposer
        self._synthesizer = synthesizer
        self._scorer = scorer
        self._rerank_k = rerank_k if rerank_k is not None else settings.rerank_k
        # NoOp default keeps ~120 existing tests untouched: they instantiate the
        # agent without supplying a tracer. Real observability is enabled by
        # the CLI/API wiring in an ``OTelTracer``.
        self._tracer: Tracer = tracer or NoOpTracer()

    async def answer(self, question: str) -> Answer:
        trace_id = new_id()
        # Bind the trace id both on structlog contextvars (every log line gets
        # it) and on ``current_trace_id`` (the retriever's spans piggy-back on
        # it via the observability contextvar).
        with request_trace(trace_id):
            started = time.perf_counter()

            async with self._tracer.stage("agent.decompose", trace_id) as span:
                sub_questions = await self._decomposer.decompose(question)
                span.set_attribute("n_sub_questions", len(sub_questions))
                _bind_component_usage(span, self._decomposer)
                log.info("decomposed", n_sub_questions=len(sub_questions))

            async with self._tracer.stage("agent.retrieve", trace_id) as span:
                hits = await self._collect_hits(sub_questions)
                span.record_hits(len(hits), hits[0].score if hits else None)
                log.info("aggregated_hits", n_hits=len(hits))

            async with self._tracer.stage("agent.synthesize", trace_id) as span:
                synthesis = await self._synthesizer.synthesize(question, hits)
                span.set_usage(synthesis.usage)
                span.set_attribute("n_citations", len(synthesis.citations))

            async with self._tracer.stage("agent.confidence", trace_id) as span:
                confidence = await self._scorer.score(question, synthesis.text, hits)
                span.set_attribute("score", float(confidence.score))
                span.set_attribute("refused", bool(confidence.refused))
                _bind_component_usage(span, self._scorer)

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

            # Stamp the provider that ACTUALLY served the synthesis step —
            # the LLM call that produced the answer text. When a live routing
            # client falls back (e.g. Anthropic unavailable → OpenAI), this
            # reflects the fallback, not the primary. Load-bearing for the
            # provenance badge in the UI.
            provider_used = synthesis.provider
            model_used = synthesis.model

            if confidence.refused:
                log.info(
                    "refused",
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
                    provider_used=provider_used,
                    model_used=model_used,
                )

            return Answer(
                query=question,
                text=synthesis.text or _REFUSAL_TEXT,
                citations=synthesis.citations,
                confidence=confidence,
                hits_used=tuple(hits),
                usage=usage,
                trace_id=trace_id,
                provider_used=provider_used,
                model_used=model_used,
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


def _bind_component_usage(span, component: object) -> None:  # type: ignore[no-untyped-def]
    """Attach a component's ``last_usage`` to the current stage span, if present."""
    usage = _last_usage_of(component)
    if usage is not None:
        span.set_usage(usage)


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
    tracer: Tracer | None = None,
) -> RagAgent:
    """Wire the default agent: LLM-based decomposer + synthesizer + scorer."""
    return RagAgent(
        retriever=retriever,
        decomposer=LLMQuestionDecomposer(client),
        synthesizer=CitedSynthesizer(client),
        scorer=LLMConfidenceScorer(client),
        tracer=tracer,
    )


__all__ = ["AgentDeps", "RagAgent", "create_default_agent"]
