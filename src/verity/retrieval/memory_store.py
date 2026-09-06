"""In-memory :class:`VectorStore` — a real backend behind the same protocol.

Two reasons this exists:

1. **Tests without a database.** The unit lane (``-m "not integration"``) must stay
   green with no Postgres running. A structural VectorStore that keeps chunks in a
   dict lets us exercise the retrieval assembly end-to-end offline.

2. **Local eval without Docker.** ``verity eval retrieval --in-memory`` produces the
   real scorecard on a laptop with no infra installed. The retriever and metrics
   don't know which backend they're talking to — the numbers are legitimate
   apples-to-apples with the pgvector run.

We use ``rank_bm25.BM25Okapi`` for the sparse channel (already a dep, mirrors what
Postgres' FTS does at a small scale) and numpy cosine for dense. Both indexes are
rebuilt on ``upsert`` — fine at the small corpus sizes this backend targets.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from verity.retrieval.base import VectorStore
from verity.types import Chunk, EmbeddedChunk, RetrievalHit, RetrieverKind

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class InMemoryVectorStore(VectorStore):
    """Dict-backed store with cosine + BM25 primitives. Not thread-safe."""

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._embeddings: dict[str, np.ndarray] = {}
        self._bm25: BM25Okapi | None = None
        self._bm25_order: list[str] = []
        self._bm25_dirty = True

    async def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        for embedded in chunks:
            cid = str(embedded.chunk.id)
            self._chunks[cid] = embedded.chunk
            self._embeddings[cid] = np.asarray(embedded.embedding, dtype=np.float32)
        self._bm25_dirty = True

    async def upsert_documents(self, documents: Sequence[Any]) -> None:
        # No document table in the in-memory backend — chunks carry everything the
        # retrieval side needs. Kept for signature compatibility with PgVectorStore.
        return None

    async def dense_search(self, query_embedding: tuple[float, ...], k: int) -> list[RetrievalHit]:
        if k <= 0 or not self._chunks:
            return []
        q = np.asarray(query_embedding, dtype=np.float32)
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0.0:
            return []

        cids = list(self._chunks.keys())
        matrix = np.stack([self._embeddings[cid] for cid in cids])
        norms = np.linalg.norm(matrix, axis=1)
        # Guard: zero-vector chunks would produce NaN — treat their similarity as -inf
        # so they never surface. In practice normalised embeddings never trigger this.
        with np.errstate(divide="ignore", invalid="ignore"):
            sims = matrix @ q / (norms * q_norm)
        sims = np.where(np.isfinite(sims), sims, -np.inf)

        top = np.argsort(-sims)[:k]
        return [
            RetrievalHit(
                chunk=self._chunks[cids[int(i)]],
                score=float(sims[int(i)]),
                kind=RetrieverKind.DENSE,
                rank=rank,
            )
            for rank, i in enumerate(top)
            if np.isfinite(sims[int(i)])
        ]

    def _rebuild_bm25(self) -> None:
        self._bm25_order = list(self._chunks.keys())
        corpus = [_tokenize(self._chunks[cid].text) for cid in self._bm25_order]
        self._bm25 = BM25Okapi(corpus) if corpus else None
        self._bm25_dirty = False

    async def sparse_search(self, query: str, k: int) -> list[RetrievalHit]:
        if k <= 0 or not query.strip():
            return []
        if self._bm25_dirty:
            self._rebuild_bm25()
        if self._bm25 is None:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = np.asarray(self._bm25.get_scores(tokens), dtype=np.float32)
        # BM25 returns 0 for docs with no match — drop them so a keyword-less query
        # doesn't spam the fusion with irrelevant chunks scored 0.
        nonzero = np.where(scores > 0)[0]
        if nonzero.size == 0:
            return []
        top = nonzero[np.argsort(-scores[nonzero])][:k]
        return [
            RetrievalHit(
                chunk=self._chunks[self._bm25_order[int(i)]],
                score=float(scores[int(i)]),
                kind=RetrieverKind.SPARSE,
                rank=rank,
            )
            for rank, i in enumerate(top)
        ]

    async def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        return [self._chunks[c] for c in chunk_ids if c in self._chunks]

    async def count_chunks(self) -> int:
        return len(self._chunks)


__all__ = ["InMemoryVectorStore"]
