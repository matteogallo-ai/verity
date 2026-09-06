"""In-memory VectorStore: dense cosine + BM25-like sparse + idempotent upsert."""

from __future__ import annotations

from uuid import UUID

import pytest

from verity.retrieval.memory_store import InMemoryVectorStore
from verity.types import Chunk, DocumentId, EmbeddedChunk, RetrieverKind


def _emb(cid: int, vec: tuple[float, ...], text: str) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk=Chunk(
            id=UUID(int=cid),
            document_id=DocumentId("00000000-0000-0000-0000-000000000000"),
            text=text,
            ordinal=cid,
            char_start=0,
            char_end=len(text),
        ),
        embedding=vec,
        model="fake",
    )


@pytest.mark.asyncio
async def test_dense_search_ranks_by_cosine() -> None:
    store = InMemoryVectorStore()
    await store.upsert(
        [
            _emb(1, (1.0, 0.0, 0.0), "alpha"),
            _emb(2, (0.0, 1.0, 0.0), "beta"),
            _emb(3, (0.7071, 0.7071, 0.0), "alphabeta"),
        ]
    )
    hits = await store.dense_search((1.0, 0.0, 0.0), k=3)
    assert [h.chunk.text for h in hits[:2]] == ["alpha", "alphabeta"]
    assert hits[0].kind is RetrieverKind.DENSE
    assert hits[0].score > hits[1].score


@pytest.mark.asyncio
async def test_sparse_search_matches_keyword() -> None:
    store = InMemoryVectorStore()
    await store.upsert(
        [
            _emb(1, (0.0, 0.0), "termination requires thirty days notice"),
            _emb(2, (0.0, 0.0), "liability is capped at one million"),
            _emb(3, (0.0, 0.0), "assignment to competitors is forbidden"),
        ]
    )
    hits = await store.sparse_search("thirty days", k=3)
    assert hits
    assert "thirty" in hits[0].chunk.text
    assert hits[0].kind is RetrieverKind.SPARSE


@pytest.mark.asyncio
async def test_sparse_search_no_match_returns_empty() -> None:
    store = InMemoryVectorStore()
    await store.upsert([_emb(1, (0.0, 0.0), "hello world")])
    assert await store.sparse_search("xyzzy", k=5) == []


@pytest.mark.asyncio
async def test_upsert_is_idempotent() -> None:
    store = InMemoryVectorStore()
    await store.upsert([_emb(1, (1.0, 0.0), "a")])
    await store.upsert([_emb(1, (1.0, 0.0), "a")])
    assert await store.count_chunks() == 1


@pytest.mark.asyncio
async def test_get_chunks_hydrates_in_order() -> None:
    store = InMemoryVectorStore()
    await store.upsert(
        [
            _emb(1, (1.0,), "one"),
            _emb(2, (1.0,), "two"),
            _emb(3, (1.0,), "three"),
        ]
    )
    got = await store.get_chunks([str(UUID(int=2)), str(UUID(int=1))])
    assert [c.text for c in got] == ["two", "one"]
