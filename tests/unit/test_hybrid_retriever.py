"""HybridRetriever end-to-end using the in-memory store + a fake embedder + reranker.

Exercises the full assembly (embed → dense ∥ sparse → RRF → rerank) without touching
Postgres or torch. Covers the resilience path where sparse returns empty.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from verity.ingestion.chunker import StructureAwareChunker
from verity.ingestion.embedder import LocalEmbedder
from verity.ingestion.parsers.markdown import MarkdownParser
from verity.retrieval import CrossEncoderReranker, HybridRetriever, InMemoryVectorStore


class _FakeEmbeddingBackend:
    """Deterministic character-based embedding — good enough to make retrieval work."""

    def __init__(self, dim: int = 8) -> None:
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
        rows: list[list[float]] = []
        for s in sentences:
            vec = [0.0] * self._dim
            for ch in s.lower():
                vec[ord(ch) % self._dim] += 1.0
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            rows.append([v / norm for v in vec])
        return rows

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim


class _FakeRerankerBackend:
    def predict(
        self,
        sentences: Sequence[tuple[str, str]],
        *,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> list[float]:
        return [float(len(set(q.lower().split()) & set(p.lower().split()))) for q, p in sentences]


async def _seed_store(tmp_path):  # type: ignore[no-untyped-def]
    md = (
        "# Contract\n\n"
        "## 1. Term\n\n"
        "Either party may terminate on thirty days notice.\n\n"
        "## 2. Liability\n\n"
        "Total liability shall not exceed one million dollars.\n\n"
        "## 3. Assignment\n\n"
        "Assignment to a competitor requires written consent.\n"
    )
    p = tmp_path / "msa.md"
    p.write_text(md, encoding="utf-8")
    doc = await MarkdownParser().parse(str(p))
    chunks = StructureAwareChunker(target_words=5, max_words=30, min_words=1).chunk(doc)
    embedder = LocalEmbedder(model="fake", expected_dim=8, backend=_FakeEmbeddingBackend(8))
    embedded = await embedder.embed_chunks(chunks)
    store = InMemoryVectorStore()
    await store.upsert(embedded)
    return store, embedder


@pytest.mark.asyncio
async def test_hybrid_retriever_returns_relevant_chunk_first(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store, embedder = await _seed_store(tmp_path)
    retriever = HybridRetriever(
        store=store,
        embedder=embedder,
        reranker=CrossEncoderReranker(model="fake", backend=_FakeRerankerBackend()),
        retrieval_k=10,
        rerank_k=3,
    )
    hits = await retriever.retrieve("terminate on thirty days notice", k=3)
    assert hits
    assert "thirty" in hits[0].chunk.text
    # Reranked hits carry the RERANKED kind + fresh 0-based ranks.
    assert hits[0].kind.value == "reranked"
    assert [h.rank for h in hits] == list(range(len(hits)))


@pytest.mark.asyncio
async def test_hybrid_retriever_degrades_when_sparse_empty(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store, embedder = await _seed_store(tmp_path)
    retriever = HybridRetriever(
        store=store,
        embedder=embedder,
        reranker=None,  # sanity: no rerank still returns hits from dense-only path
        retrieval_k=5,
        rerank_k=3,
    )
    # Query with only tokens absent from the corpus → sparse returns []; dense saves it.
    hits = await retriever.retrieve("qxyz vvvv", k=3)
    assert hits, "retriever must still return dense hits when sparse yields nothing"
