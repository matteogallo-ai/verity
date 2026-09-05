"""Embedder shape, dim guard, and async wrapping — with a fake backend, no torch."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from verity.ingestion.embedder import LocalEmbedder
from verity.types import Chunk, DocumentId


class _FakeBackend:
    def __init__(self, dim: int) -> None:
        self._dim = dim

    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> list[list[float]]:
        return [[float(len(s) % 7) / 7.0] * self._dim for s in sentences]

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim


@pytest.mark.asyncio
async def test_embed_returns_vectors_of_expected_dim() -> None:
    embedder = LocalEmbedder(model="fake", expected_dim=8, backend=_FakeBackend(8))
    vectors = await embedder.embed(["hello", "world"])
    assert len(vectors) == 2
    assert all(len(v) == 8 for v in vectors)


@pytest.mark.asyncio
async def test_embed_empty_input_short_circuits() -> None:
    embedder = LocalEmbedder(model="fake", expected_dim=8, backend=_FakeBackend(8))
    assert await embedder.embed([]) == []
    assert await embedder.embed_chunks([]) == []


@pytest.mark.asyncio
async def test_dim_mismatch_raises_clearly() -> None:
    embedder = LocalEmbedder(model="fake", expected_dim=384, backend=_FakeBackend(8))
    with pytest.raises(RuntimeError, match="dim=8"):
        await embedder.embed(["x"])


@pytest.mark.asyncio
async def test_embed_chunks_binds_model_id() -> None:
    embedder = LocalEmbedder(model="fake-v1", expected_dim=4, backend=_FakeBackend(4))
    doc_id: DocumentId = __import__("uuid").uuid4()
    chunks = [
        Chunk(document_id=doc_id, text="a", ordinal=0, char_start=0, char_end=1),
        Chunk(document_id=doc_id, text="bc", ordinal=1, char_start=1, char_end=3),
    ]
    out = await embedder.embed_chunks(chunks)
    assert [e.chunk for e in out] == chunks
    assert all(e.model == "fake-v1" for e in out)
    assert all(len(e.embedding) == 4 for e in out)
