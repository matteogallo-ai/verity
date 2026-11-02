"""Top-level agentic RAG orchestrator.

Pipeline for one ``answer(question)`` call:

1. **Decompose** the question into ordered sub-questions (single element if simple).
2. **Retrieve** per sub-question through the hybrid retriever; deduplicate hits by
   chunk id while preserving the highest-scoring occurrence.
3. **Synthesize** a draft answer + validated citations grounded in the aggregate
   evidence.
4. **Score** confidence and refuse below-threshold rather than shipping a
   fabricated answer.

What holds on the returned :class:`Answer`:
- either ``error`` is set (a provider or the retrieval store failed; no answer,
  no refusal) or ``confidence`` is;
- every citation's ``quote`` is a verbatim substring of the referenced chunk
  (citations that are not are dropped in :class:`CitedSynthesizer`), and a
  non-refused answer carries at least one: a draft with none is refused with
  reason ``no_citation`` before the scorer runs;
- ``confidence.refused`` is also True when the scorer model asked to refuse OR its
  self-reported score is below ``confidence_threshold`` (threshold-gated refusal;
  the score is not calibrated). A refused answer's text is a fixed "I don't know"
  with the rationale;
- ``hits_used`` and ``usage`` are populated for observability and eval.
"""

from __future__ import annotations

import time

import structlog

from verity.agent.base import Agent, ConfidenceScorer, QuestionDecomposer
from verity.agent.decomposer import LLMQuestionDecomposer
from verity.agent.scorer import LLMConfidenceScorer
from verity.agent.synthesizer import CitedSynthesizer
from verity.config import get_settings
from verity.llm.base import (
    AuthenticationError,
    LLMError,
    ProviderUnavailableError,
    RoutingClient,
)
from verity.llm.routing import primary_client_identity
from verity.observability.base import Tracer
from verity.observability.logging import request_trace
from verity.observability.tracer import NoOpTracer
from verity.retrieval.base import RetrievalUnavailableError, Retriever
from verity.types import (
    NO_CITATION_RATIONALE,
    REFUSAL_TEXT,
    Answer,
    AnswerError,
    Confidence,
    ErrorKind,
    Provider,
    RefusalReason,
    RetrievalHit,
    TraceId,
    UsageStats,
    new_id,
)

log = structlog.get_logger(__name__)


class RagAgent(Agent):
    """The public entry point the CLI and the API call."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        decomposer: QuestionDecomposer,
        synthesizer: CitedSynthesizer,
        scorer: ConfidenceScorer,
        rerank_k: int | None = None,
        tracer: Tracer | None = None,
        preferred_provider: Provider | None = None,
        preferred_model: str | None = None,
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
        # Routing-decision fallback stamped on every Answer this agent emits.
        # Prevents a synthesizer that skips the router from producing an Answer with
        # ``provider_used = None`` and letting downstream provenance fall back
        # to a boot-time default that lies about who would have served.
        self._preferred_provider = preferred_provider
        self._preferred_model = preferred_model

    async def answer(self, question: str) -> Answer:
        trace_id = new_id()
        # Bind the trace id both on structlog contextvars (every log line gets
        # it) and on ``current_trace_id`` (the retriever's spans piggy-back on
        # it via the observability contextvar).
        with request_trace(trace_id):
            started = time.perf_counter()
            # An infrastructure failure is returned as a typed error, never as a
            # refusal: "the provider is down" must not read as "the evidence is weak".
            try:
                return await self._answer(question, trace_id, started)
            except RetrievalUnavailableError as exc:
                return _error_answer(
                    question, trace_id, started, ErrorKind.RETRIEVAL_UNAVAILABLE, exc
                )
            except AuthenticationError as exc:
                return _error_answer(question, trace_id, started, ErrorKind.PROVIDER_AUTH, exc)
            except ProviderUnavailableError as exc:
                return _error_answer(
                    question, trace_id, started, ErrorKind.PROVIDER_UNAVAILABLE, exc
                )
            except LLMError as exc:
                return _error_answer(question, trace_id, started, ErrorKind.PROVIDER_ERROR, exc)

    async def _answer(self, question: str, trace_id: TraceId, started: float) -> Answer:
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

        scorer_usage: UsageStats | None = None
        async with self._tracer.stage("agent.confidence", trace_id) as span:
            if not synthesis.citations or not synthesis.text.strip():
                # Deterministic gate before the scorer: a draft with no verified
                # citation is refused whatever a model would say about it.
                confidence = Confidence(
                    score=0.0,
                    refused=True,
                    rationale=NO_CITATION_RATIONALE,
                    reason=RefusalReason.NO_CITATION,
                )
            else:
                confidence = await self._scorer.score(question, synthesis.text, hits)
                scorer_usage = _last_usage_of(self._scorer)
                _bind_component_usage(span, self._scorer)
            span.set_attribute("score", float(confidence.score))
            span.set_attribute("refused", bool(confidence.refused))

        # Latency is the measured wall-clock, not a sum of per-call latencies:
        # the calls partially overlap, so summing would over-count.
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        usage = _aggregate_usage(
            _last_usage_of(self._decomposer),
            synthesis.usage,
            scorer_usage,
            latency_ms=latency_ms,
        )

        # Stamp the provider that ACTUALLY served the synthesis step, so a
        # routing fallback (e.g. Anthropic unavailable → OpenAI) is reported as
        # such. If a synthesizer returns without consulting the router, fall back
        # to the routing decision stamped at construction, never None.
        provider_used = synthesis.provider or self._preferred_provider
        model_used = synthesis.model or self._preferred_model

        if confidence.refused:
            log.info(
                "refused",
                score=confidence.score,
                reason=confidence.reason,
                rationale=confidence.rationale,
            )
            return Answer(
                query=question,
                text=REFUSAL_TEXT,
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
            text=synthesis.text,
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

        Sub-questions are retrieved serially rather than with gather() so log lines
        and per-stage spans interleave in question order, which keeps a trace
        readable. Typical questions decompose into one or two sub-questions, so the
        latency cost is small.
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


def _error_answer(
    question: str, trace_id: TraceId, started: float, kind: ErrorKind, exc: Exception
) -> Answer:
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    log.error("answer_failed", error_kind=kind.value, error=str(exc), latency_ms=latency_ms)
    return Answer(
        query=question,
        text="",
        error=AnswerError(kind=kind, detail=f"{type(exc).__name__}: {exc}"),
        usage=UsageStats(latency_ms=latency_ms),
        trace_id=trace_id,
    )


def _bind_component_usage(span, component: object) -> None:  # type: ignore[no-untyped-def]
    """Attach a component's ``last_usage`` to the current stage span, if present."""
    usage = _last_usage_of(component)
    if usage is not None:
        span.set_usage(usage)


def _last_usage_of(component: object) -> UsageStats | None:
    """Best-effort accessor for the ``last_usage`` attribute the concrete LLM-backed
    decomposer / scorer expose. Protocol implementations without it contribute nothing
    to the aggregate rather than a fabricated count."""
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
    """Wire the default agent: LLM-based decomposer + synthesizer + scorer.

    The routing client's first configured provider + model is stamped on the
    agent as the ``preferred_*`` fallback so any Answer (including one whose
    synthesizer skipped the router) carries the real
    routing decision — not a None that would leak up to the API's boot-time
    default and lie about who would have served.
    """
    preferred_provider, preferred_model = primary_client_identity(client)
    return RagAgent(
        retriever=retriever,
        decomposer=LLMQuestionDecomposer(client),
        synthesizer=CitedSynthesizer(client),
        scorer=LLMConfidenceScorer(client),
        tracer=tracer,
        preferred_provider=preferred_provider,
        preferred_model=preferred_model,
    )


__all__ = ["RagAgent", "create_default_agent"]
