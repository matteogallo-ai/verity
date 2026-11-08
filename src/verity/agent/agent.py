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
- a cited draft is also refused before the scorer when its quotes cover too little
  of it, or, with ``irrelevant_citation_check`` set, when the reranker judged them
  irrelevant (:mod:`verity.agent.support`), so a document that steers the scorer
  cannot ship an answer on its own;
- ``confidence.refused`` is also True when the scorer model asked to refuse OR its
  self-reported score is below ``confidence_threshold`` (the score is the model's
  self-report and is not calibrated; at the default threshold, 0, no score is below it). A refused answer's text is a fixed "I don't know"
  with the rationale;
- ``hits_used`` and ``usage`` are populated for observability and eval.

With ``checks=False`` the agent is the naive RAG baseline the eval compares against
(docs/preregistration/naive-rag-baseline.md): the draft ships as the synthesizer wrote
it, with no citation gate, no support check, no scorer call and no threshold. Only the
synthesizer's verbatim filter remains, because a citation must point at a span.
"""

from __future__ import annotations

import math
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog

from verity.agent.base import Agent, ConfidenceScorer, QuestionDecomposer
from verity.agent.decomposer import LLMQuestionDecomposer
from verity.agent.scorer import LLMConfidenceScorer
from verity.agent.support import check_support
from verity.agent.synthesizer import CitedSynthesizer, SynthesisResult
from verity.config import get_settings
from verity.llm.base import (
    AuthenticationError,
    LLMError,
    ProviderUnavailableError,
    RoutingClient,
)
from verity.llm.budget import SpendCapExceededError
from verity.llm.routing import primary_client_identity
from verity.observability.base import StageSpan, Tracer
from verity.observability.logging import request_trace
from verity.observability.tracer import NoOpTracer
from verity.retrieval.base import ModelUnavailableError, RetrievalUnavailableError, Retriever
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

# The naive baseline never scores: 1.0 records "shipped without a gate", not a belief.
NAIVE_CONFIDENCE = Confidence(
    score=1.0, refused=False, rationale="checks disabled: naive RAG baseline, draft shipped"
)


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
        checks: bool = True,
    ) -> None:
        settings = get_settings()
        self._checks = checks
        self._retriever = retriever
        self._decomposer = decomposer
        self._synthesizer = synthesizer
        self._scorer = scorer
        self._rerank_k = rerank_k if rerank_k is not None else settings.rerank_k
        self._min_citation_coverage = settings.min_citation_coverage
        # A floor of -inf turns the cited relevance check off without a second code
        # path in verity.agent.support: no score is below it.
        self._min_cited_rerank_score = (
            settings.min_cited_rerank_score if settings.irrelevant_citation_check else -math.inf
        )
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
            # Filled as each stage ends, so an error line shows how far the request got.
            stage_ms: dict[str, float] = {}
            # Usage of each LLM call that returned, so a failure later in the request
            # still reports what it cost: spending caps count errored answers too.
            spent: list[UsageStats] = []
            error: Exception
            # An infrastructure failure is returned as a typed error, never as a
            # refusal: "the provider is down" must not read as "the evidence is weak".
            try:
                answer = await self._answer(question, trace_id, started, stage_ms, spent)
            except ModelUnavailableError as exc:
                kind, error = ErrorKind.MODEL_UNAVAILABLE, exc
            except RetrievalUnavailableError as exc:
                kind, error = ErrorKind.RETRIEVAL_UNAVAILABLE, exc
            except SpendCapExceededError as exc:
                kind, error = ErrorKind.BUDGET_EXCEEDED, exc
            except AuthenticationError as exc:
                kind, error = ErrorKind.PROVIDER_AUTH, exc
            except ProviderUnavailableError as exc:
                kind, error = ErrorKind.PROVIDER_UNAVAILABLE, exc
            except LLMError as exc:
                kind, error = ErrorKind.PROVIDER_ERROR, exc
            # A defect anywhere in the pipeline. Caught only to keep what the request
            # has already paid for (spending budgets count it) and to log the
            # traceback; it is typed INTERNAL, so it never reads as an outage.
            except Exception as exc:
                log.error("answer_defect", trace_id=str(trace_id), exc_info=exc)
                kind, error = ErrorKind.INTERNAL, exc
            else:
                _log_answer(answer, stage_ms)
                return answer
            return _error_answer(question, trace_id, started, kind, error, stage_ms, spent)

    @asynccontextmanager
    async def _stage(
        self, name: str, trace_id: TraceId, stage_ms: dict[str, float]
    ) -> AsyncIterator[StageSpan]:
        """A tracer span that also records its wall-clock time in ``stage_ms``."""
        stage_started = time.perf_counter()
        try:
            async with self._tracer.stage(name, trace_id) as span:
                yield span
        finally:
            stage_ms[name] = round((time.perf_counter() - stage_started) * 1000, 2)

    async def _answer(
        self,
        question: str,
        trace_id: TraceId,
        started: float,
        stage_ms: dict[str, float],
        spent: list[UsageStats],
    ) -> Answer:
        async with self._stage("agent.decompose", trace_id, stage_ms) as span:
            sub_questions = await self._decomposer.decompose(question)
            span.set_attribute("n_sub_questions", len(sub_questions))
            _record_usage(span, spent, _last_usage_of(self._decomposer))
            log.info("decomposed", n_sub_questions=len(sub_questions))

        async with self._stage("agent.retrieve", trace_id, stage_ms) as span:
            hits = await self._collect_hits(sub_questions)
            span.record_hits(len(hits), hits[0].score if hits else None)
            log.info("aggregated_hits", n_hits=len(hits))

        async with self._stage("agent.synthesize", trace_id, stage_ms) as span:
            synthesis = await self._synthesizer.synthesize(question, hits)
            _record_usage(span, spent, synthesis.usage)
            span.set_attribute("n_citations", len(synthesis.citations))

        if not self._checks:
            return self._shipped(
                question, synthesis, hits, NAIVE_CONFIDENCE, trace_id, started, stage_ms, spent
            )

        async with self._stage("agent.confidence", trace_id, stage_ms) as span:
            # Deterministic gates before the scorer: a draft with no verified citation,
            # or whose citations do not cover it or are irrelevant to the question, is
            # refused whatever a model reading the same documents would say about it.
            support = check_support(
                synthesis.text,
                synthesis.citations,
                hits,
                min_coverage=self._min_citation_coverage,
                min_rerank_score=self._min_cited_rerank_score,
            )
            if not synthesis.citations or not synthesis.text.strip():
                confidence = Confidence(
                    score=0.0,
                    refused=True,
                    rationale=NO_CITATION_RATIONALE,
                    reason=RefusalReason.NO_CITATION,
                )
            elif support.reason is not None:
                confidence = Confidence(
                    score=0.0, refused=True, rationale=support.rationale, reason=support.reason
                )
            else:
                confidence = await self._scorer.score(question, synthesis.text, hits)
                _record_usage(span, spent, _last_usage_of(self._scorer))
            span.set_attribute("score", float(confidence.score))
            span.set_attribute("refused", bool(confidence.refused))
            span.set_attribute("citation_coverage", support.coverage)

        return self._shipped(
            question, synthesis, hits, confidence, trace_id, started, stage_ms, spent
        )

    def _shipped(
        self,
        question: str,
        synthesis: SynthesisResult,
        hits: list[RetrievalHit],
        confidence: Confidence,
        trace_id: TraceId,
        started: float,
        stage_ms: dict[str, float],
        spent: list[UsageStats],
    ) -> Answer:
        # Latency is the measured wall-clock, not a sum of per-call latencies:
        # the calls partially overlap, so summing would over-count.
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        usage = _aggregate_usage(*spent, latency_ms=latency_ms)

        # Stamp the provider that ACTUALLY served the synthesis step, so a
        # routing fallback (e.g. Anthropic unavailable → OpenAI) is reported as
        # such. If a synthesizer returns without consulting the router, fall back
        # to the routing decision stamped at construction, never None.
        provider_used = synthesis.provider or self._preferred_provider
        model_used = synthesis.model or self._preferred_model

        if confidence.refused:
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
                stage_ms=dict(stage_ms),
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
            stage_ms=dict(stage_ms),
            checks_disabled=not self._checks,
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


def _log_answer(answer: Answer, stage_ms: dict[str, float]) -> None:
    """The one line per answered or refused request: what operations reads first."""
    confidence = answer.confidence
    assert confidence is not None and answer.usage is not None
    log.info(
        "answer_done",
        trace_id=str(answer.trace_id),
        decision="refused" if confidence.refused else "answered",
        reason=confidence.reason.value if confidence.reason else None,
        score=confidence.score,
        rationale=confidence.rationale if confidence.refused else None,
        stage_ms=stage_ms,
        latency_ms=answer.usage.latency_ms,
        input_tokens=answer.usage.input_tokens,
        output_tokens=answer.usage.output_tokens,
        cost_usd=answer.usage.cost_usd,
        n_citations=len(answer.citations),
    )


def _error_answer(
    question: str,
    trace_id: TraceId,
    started: float,
    kind: ErrorKind,
    exc: Exception,
    stage_ms: dict[str, float],
    spent: list[UsageStats],
) -> Answer:
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    usage = _aggregate_usage(*spent, latency_ms=latency_ms)
    log.error(
        "answer_failed",
        trace_id=str(trace_id),
        decision="error",
        error_kind=kind.value,
        error=str(exc),
        stage_ms=stage_ms,
        latency_ms=latency_ms,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=usage.cost_usd,
    )
    return Answer(
        query=question,
        text="",
        error=AnswerError(kind=kind, detail=f"{type(exc).__name__}: {exc}"),
        usage=usage,
        trace_id=trace_id,
        stage_ms=dict(stage_ms),
    )


def _record_usage(span: StageSpan, spent: list[UsageStats], usage: UsageStats | None) -> None:
    """Attach a stage's usage to its span and to the request's running total."""
    if usage is not None:
        span.set_usage(usage)
        spent.append(usage)


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
    retries = 0
    for part in parts:
        if part is None:
            continue
        input_tokens += part.input_tokens
        output_tokens += part.output_tokens
        llm_calls += part.llm_calls
        cost_usd += part.cost_usd
        retries += part.retries
    return UsageStats(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        llm_calls=llm_calls,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        retries=retries,
    )


def create_default_agent(
    *,
    retriever: Retriever,
    client: RoutingClient,
    tracer: Tracer | None = None,
    checks: bool = True,
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
        checks=checks,
    )


__all__ = ["RagAgent", "create_default_agent"]
