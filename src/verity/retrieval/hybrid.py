"""Hybrid retrieval — embed → dense ∥ sparse → RRF → cross-encoder rerank.

The single entry point the agent (S3) will depend on. Every stage speaks the
:class:`RetrievalHit` type from ``verity.types`` and is exposed as its own protocol,
so swapping any stage — a different vector store, a different fusion, no reranker —
is one constructor argument away and doesn't ripple.

Stage-level observability lands here through structlog: each stage emits a single
``stage`` event with count + top score. When S5 introduces the tracer contract this
module becomes the natural place to bind spans; the log surface stays the same so
downstream diffs are noise-free.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable

import structlog

from verity.config import get_settings
from verity.ingestion.base import Embedder
from verity.retrieval.base import Fusion, Reranker, Retriever, VectorStore
from verity.retrieval.fusion import ReciprocalRankFusion
from verity.types import RetrievalHit

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
    ) -> None:
        settings = get_settings()
        self._store = store
        self._embedder = embedder
        self._reranker = reranker
        self._fusion: Fusion = fusion or ReciprocalRankFusion()
        self._retrieval_k = retrieval_k if retrieval_k is not None else settings.retrieval_k
        self._rerank_k = rerank_k if rerank_k is not None else settings.rerank_k

    async def retrieve(self, query: str, k: int) -> list[RetrievalHit]:
        """Return the final ranked evidence set for ``query``.

        ``k`` caps the returned list. If a reranker is present, ``rerank_k`` (from
        config or constructor) also caps what the reranker keeps, and the final list
        is ``min(k, rerank_k)``. When ``k`` is smaller than ``retrieval_k``, dense
        and sparse are still fetched at ``retrieval_k`` — recall at fusion time
        matters more than avoiding a few extra DB rows.
        """
        if k <= 0:
            return []

        started = time.perf_counter()
        query_vec = (await self._embedder.embed([query]))[0]

        # Run dense and sparse concurrently — both are IO-bound against Postgres.
        dense_task = asyncio.create_task(
            self._safe_stage(
                "dense",
                self._store.dense_search(query_vec, self._retrieval_k),
            )
        )
        sparse_task = asyncio.create_task(
            self._safe_stage(
                "sparse",
                self._store.sparse_search(query, self._retrieval_k),
            )
        )
        dense_hits, sparse_hits = await asyncio.gather(dense_task, sparse_task)

        fused = self._fusion.fuse([dense_hits, sparse_hits], k=max(self._retrieval_k, k))
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

        reranked = await self._reranker.rerank(query, fused, top_k=min(self._rerank_k, k))
        log.info(
            "stage",
            stage="reranked",
            count=len(reranked),
            top_score=reranked[0].score if reranked else None,
        )
        self._log_final(reranked, started)
        return reranked

    async def _safe_stage(
        self, name: str, coro: Awaitable[list[RetrievalHit]]
    ) -> list[RetrievalHit]:
        """Run one retrieval stage, log its outcome, and degrade to [] on failure.

        Robustness rule from the S2 spec: sparse returning nothing (word absent from
        FTS) or a stage error must not crash the whole retrieve() — the other stage
        can still produce hits. The error is logged loudly, never swallowed silently.
        """
        try:
            hits = await coro
        except Exception as exc:  # intentional broad catch: log + degrade to []
            log.error("stage_failed", stage=name, error=str(exc))
            return []
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
