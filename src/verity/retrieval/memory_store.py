"""In-memory :class:`VectorStore` — a real backend behind the same protocol.

Two reasons this exists:

1. **Tests without a database.** The unit lane (``-m "not integration"``) must stay
   green with no Postgres running. A structural VectorStore that keeps chunks in a
   dict lets us exercise the retrieval assembly end-to-end offline.

2. **Local runs without Docker.** ``VERITY_STORE=memory`` runs the CLI, the API and
   the eval with no database.

The sparse channel is ``rank_bm25.BM25Okapi``; Postgres uses ``ts_rank`` on English
lexemes. They are different scorers and do not rank identically: on the eval corpus
both put every gold chunk in their top 5 and agree on the top hit for 17 of 19
questions (``tests/integration/test_sparse_parity.py``). Numbers from the two stores
are therefore close, not interchangeable. Dense is numpy cosine. Both indexes are
rebuilt on ``upsert``, fine at the small corpus sizes this backend targets.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from verity.ingestion.ids import content_sha256
from verity.retrieval.base import CorpusStore, DocumentSummary, IndexedDocument
from verity.types import Chunk, Document, EmbeddedChunk, RetrievalHit, RetrieverKind

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class InMemoryVectorStore(CorpusStore):
    """Dict-backed store with cosine + BM25 primitives. Not thread-safe."""

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._documents: dict[str, Document] = {}
        self._embeddings: dict[str, np.ndarray] = {}
        self._models: dict[str, str] = {}
        self._bm25: BM25Okapi | None = None
        self._bm25_order: list[str] = []
        self._bm25_dirty = True

    async def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        for embedded in chunks:
            cid = str(embedded.chunk.id)
            self._chunks[cid] = embedded.chunk
            self._embeddings[cid] = np.asarray(embedded.embedding, dtype=np.float32)
            self._models[cid] = embedded.model
        self._bm25_dirty = True

    async def indexed_document(self, document_id: str) -> IndexedDocument | None:
        document = self._documents.get(document_id)
        if document is None:
            return None
        cids = self._chunk_ids_of(document_id)
        return IndexedDocument(
            content_sha256=content_sha256(document.text),
            chunk_ids=frozenset(cids),
            embedding_models=frozenset(self._models[c] for c in cids),
        )

    async def replace_document(self, document: Document, chunks: list[EmbeddedChunk]) -> None:
        # Nothing awaits between the removal and the insert, so no reader sees the gap.
        self._drop_chunks(str(document.id))
        self._documents[str(document.id)] = document
        await self.upsert(chunks)

    async def delete_documents(self, document_ids: Sequence[str]) -> None:
        for document_id in document_ids:
            self._drop_chunks(document_id)
            self._documents.pop(document_id, None)

    def _chunk_ids_of(self, document_id: str) -> list[str]:
        return [cid for cid, c in self._chunks.items() if str(c.document_id) == document_id]

    def _drop_chunks(self, document_id: str) -> None:
        for cid in self._chunk_ids_of(document_id):
            del self._chunks[cid]
            del self._embeddings[cid]
            del self._models[cid]
        self._bm25_dirty = True

    async def list_documents(self) -> list[DocumentSummary]:
        counts: dict[str, int] = {}
        for chunk in self._chunks.values():
            key = str(chunk.document_id)
            counts[key] = counts.get(key, 0) + 1
        return [
            DocumentSummary(
                document_id=key, uri=doc.uri, title=doc.title, n_chunks=counts.get(key, 0)
            )
            for key, doc in sorted(self._documents.items(), key=lambda kv: kv[1].uri)
        ]

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
