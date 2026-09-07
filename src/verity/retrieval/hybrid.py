"""Hybrid retrieval — embed → dense ∥ sparse → RRF → cross-encoder rerank.

The single entry point the agent (S3) depends on. Every stage speaks the
:class:`RetrievalHit` type from ``verity.types`` and is exposed as its own protocol,
so swapping any stage — a different vector store, a different fusion, no reranker —
is one constructor argument away and doesn't ripple.

Observability (S5): each stage is bracketed by a span from the injected
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
from verity.retrieval.base import Fusion, Reranker, Retriever, VectorStore
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
            query_vec = (await self._embedder.embed([query]))[0]

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
            reranked = await self._reranker.rerank(query, fused, top_k=min(self._rerank_k, k))
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
        """Run one retrieval stage, log its outcome, and degrade to [] on failure.

        Robustness rule from S2: sparse returning nothing (word absent from FTS)
        or a stage error must not crash the whole retrieve() — the other stage
        can still produce hits. The error is logged loudly, never swallowed.
        """
        async with self._tracer.stage(name, trace_id) as span:
            try:
                hits = await coro
            except Exception as exc:  # intentional broad catch: log + degrade to []
                log.error("stage_failed", stage=name, error=str(exc))
                span.set_attribute("error", str(exc))
                return []
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


__all__ = ["HybridRetriever"]
