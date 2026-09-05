"""Retrieval contracts: dense + sparse retrievers, RRF fusion, cross-encoder rerank.

The pipeline is deliberately staged and each stage is a swappable protocol so the eval
harness can score them independently (retrieval precision/recall is measured on the
fused+reranked set, but a regression can be localised to a single stage).

Implementations land in S2.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from verity.types import Chunk, EmbeddedChunk, RetrievalHit


@runtime_checkable
class VectorStore(Protocol):
    """Persistence for embedded chunks + dense/sparse search. Backed by pgvector.

    A single store owns both the dense index (pgvector) and the sparse index (Postgres
    FTS / BM25) so that ``docker-compose up`` needs exactly one stateful service.
    """

    async def upsert(self, chunks: list[EmbeddedChunk]) -> None: ...

    async def dense_search(
        self, query_embedding: tuple[float, ...], k: int
    ) -> list[RetrievalHit]: ...

    async def sparse_search(self, query: str, k: int) -> list[RetrievalHit]: ...

    async def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]: ...


@runtime_checkable
class Retriever(Protocol):
    """Full hybrid retrieval for a query: dense + sparse, fused, reranked.

    Returns the final ordered evidence set the agent reasons over. The stage-level
    hits (dense-only, sparse-only, fused) remain observable via the tracer for debugging
    and eval, but this is the single entry point the agent depends on.
    """

    async def retrieve(self, query: str, k: int) -> list[RetrievalHit]: ...


@runtime_checkable
class Fusion(Protocol):
    """Combines several ranked lists into one. Reciprocal Rank Fusion by default."""

    def fuse(self, ranked_lists: list[list[RetrievalHit]], k: int) -> list[RetrievalHit]: ...


@runtime_checkable
class Reranker(Protocol):
    """Re-scores candidate hits against the query with a cross-encoder.

    Runs locally (offline) by default so CI and the eval harness need no API keys.
    """

    async def rerank(
        self, query: str, hits: list[RetrievalHit], top_k: int
    ) -> list[RetrievalHit]: ...
