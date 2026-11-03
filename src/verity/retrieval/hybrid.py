"""Hybrid retrieval — embed → dense ∥ sparse → RRF → cross-encoder rerank.

The single retrieval entry point the agent depends on. Every stage speaks the
:class:`RetrievalHit` type from ``verity.types`` and is exposed as its own protocol,
so swapping any stage — a different vector store, a different fusion, no reranker —
is one constructor argument away and doesn't ripple.

Observability: each stage is bracketed by a span from the injected
:class:`Tracer`. The default :class:`NoOpTracer` makes the instrumentation free
for callers that don't want it (existing unit tests, one-off scripts). Passing
an :class:`OTelTracer` in from the CLI / API surfaces the per-stage latency +
retrieval hit counts under the query's ``trace_id``.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable

import structlog

from verity.config import get_settings
from verity.ingestion.base import Embedder
from verity.observability.base import Tracer
from verity.observability.logging import current_trace_id
from verity.observability.tracer import NoOpTracer
from verity.retrieval.base import (
    Fusion,
    ModelUnavailableError,
    Reranker,
    RetrievalUnavailableError,
    Retriever,
    VectorStore,
)
from verity.retrieval.fusion import ReciprocalRankFusion
from verity.types import RetrievalHit, TraceId, new_id

log = structlog.get_logger(__name__)


class HybridRetriever(Retriever):
    """Full hybrid pipeline. Construct with the store + embedder + (optional) reranker."""

    def __init__(
        self,
        *,
        store: VectorStore,
        embedder: Embedder,
        reranker: Reranker | None = None,
        fusion: Fusion | None = None,
        retrieval_k: int | None = None,
        rerank_k: int | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        settings = get_settings()
        self._store = store
        self._embedder = embedder
        self._reranker = reranker
        self._fusion: Fusion = fusion or ReciprocalRankFusion()
        self._retrieval_k = retrieval_k if retrieval_k is not None else settings.retrieval_k
        self._rerank_k = rerank_k if rerank_k is not None else settings.rerank_k
        self._tracer: Tracer = tracer or NoOpTracer()

    async def retrieve(self, query: str, k: int) -> list[RetrievalHit]:
        """Return the final ranked evidence set for ``query``.

        The trace id spans piggy-back on is read from
        :data:`verity.observability.logging.current_trace_id` — set by the
        agent at the top of ``answer()``. If nothing is set (direct callers
        that don't bind a trace), we generate one so exported spans still
        have a group id.
        """
        if k <= 0:
            return []

        span_trace_id: TraceId = current_trace_id.get() or new_id()
        started = time.perf_counter()

        async with self._tracer.stage("retrieve.embed_query", span_trace_id) as span:
            span.set_attribute("query.length", len(query))
            query_vec = (await _model_call("embedder", self._embedder.embed([query])))[0]

        # Run dense and sparse concurrently — both are IO-bound against Postgres.
        dense_task = asyncio.create_task(
            self._safe_stage(
                "retrieve.dense",
                self._store.dense_search(query_vec, self._retrieval_k),
                trace_id=span_trace_id,
            )
        )
        sparse_task = asyncio.create_task(
            self._safe_stage(
                "retrieve.sparse",
                self._store.sparse_search(query, self._retrieval_k),
                trace_id=span_trace_id,
            )
        )
        dense_hits, sparse_hits = await asyncio.gather(dense_task, sparse_task)

        async with self._tracer.stage("retrieve.fuse", span_trace_id) as span:
            fused = self._fusion.fuse([dense_hits, sparse_hits], k=max(self._retrieval_k, k))
            span.record_hits(len(fused), fused[0].score if fused else None)
            log.info(
                "stage",
                stage="fused",
                count=len(fused),
                top_score=fused[0].score if fused else None,
            )

        if self._reranker is None:
            final = fused[:k]
            self._log_final(final, started)
            return final

        async with self._tracer.stage("retrieve.rerank", span_trace_id) as span:
            reranked = await _model_call(
                "reranker", self._reranker.rerank(query, fused, top_k=min(self._rerank_k, k))
            )
            span.record_hits(len(reranked), reranked[0].score if reranked else None)
            log.info(
                "stage",
                stage="reranked",
                count=len(reranked),
                top_score=reranked[0].score if reranked else None,
            )
        self._log_final(reranked, started)
        return reranked

    async def _safe_stage(
        self,
        name: str,
        coro: Awaitable[list[RetrievalHit]],
        *,
        trace_id: TraceId,
    ) -> list[RetrievalHit]:
        """Run one retrieval stage and log its outcome.

        An empty stage (no keyword match in full-text search) is a valid result. A
        stage that raises (store down, query error) is not: degrading it to ``[]``
        would make an outage look like weak evidence and turn it into a refusal, so
        it is raised as :class:`RetrievalUnavailableError`.
        """
        async with self._tracer.stage(name, trace_id) as span:
            try:
                hits = await coro
            # Any store or driver exception is re-raised as the typed error the
            # agent maps to an infrastructure failure.
            except Exception as exc:
                log.error("stage_failed", stage=name, error=str(exc))
                span.set_attribute("error", str(exc))
                raise RetrievalUnavailableError(f"{name} failed: {exc}") from exc
            span.record_hits(len(hits), hits[0].score if hits else None)
            log.info(
                "stage",
                stage=name,
                count=len(hits),
                top_score=hits[0].score if hits else None,
            )
            return hits

    def _log_final(self, final: list[RetrievalHit], started: float) -> None:
        log.info(
            "retrieve_done",
            n_hits=len(final),
            elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
        )


async def _model_call[T](name: str, coro: Awaitable[T]) -> T:
    """Await a local model call; any failure becomes :class:`ModelUnavailableError`.

    Loading or running a model raises whatever the Hugging Face cache, torch or the OS
    raise (``OSError``, ``RuntimeError``, ``ValueError`` …). Left untyped, it reaches
    the API as a 500 and the agent cannot report the LLM spend already made."""
    try:
        return await coro
    except Exception as exc:
        log.error("model_failed", model=name, error=str(exc))
        raise ModelUnavailableError(f"{name} failed: {exc}") from exc


__all__ = ["HybridRetriever"]
