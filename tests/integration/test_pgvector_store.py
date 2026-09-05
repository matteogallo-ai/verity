"""Integration: migrations + PgVectorStore round-trip against a live pgvector db.

Marked @pytest.mark.integration; the ``not integration`` CI lane skips it. The
integration lane in CI provisions a pgvector service and sets ``VERITY_DATABASE_URL``
so this test runs end-to-end.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import pytest

from verity.db import apply_pending
from verity.ingestion import (
    IngestionPipeline,
    StructureAwareChunker,
    create_default_parser,
)
from verity.ingestion.embedder import LocalEmbedder
from verity.ingestion.pipeline import discover_sources
from verity.retrieval import PgVectorStore

pytestmark = pytest.mark.integration


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
        # Deterministic per-text vector so dense_search has a signal to rank.
        out: list[list[float]] = []
        for s in sentences:
            base = float(sum(ord(c) for c in s) % 97) / 97.0
            out.append([base] * self._dim)
        return out

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim


def _database_url() -> str:
    return os.environ.get("VERITY_DATABASE_URL", "postgresql://verity:verity@localhost:5432/verity")


@pytest.fixture(scope="module")
def migrated_db() -> str:
    url = _database_url()
    apply_pending(url)
    return url


@pytest.mark.asyncio
async def test_upsert_dense_sparse_roundtrip(migrated_db: str, tmp_path) -> None:  # type: ignore[no-untyped-def]
    # Build a tiny in-tmp corpus so we don't collide with the shipped example dataset.
    (tmp_path / "a.md").write_text(
        "# Termination\n\nEither party may terminate on thirty (30) days notice.\n"
    )
    (tmp_path / "b.md").write_text(
        "# Liability\n\nAggregate liability shall not exceed one million dollars.\n"
    )
    parser = create_default_parser()
    chunker = StructureAwareChunker()
    embedder = LocalEmbedder(model="fake", expected_dim=384, backend=_FakeBackend(384))
    store = PgVectorStore(database_url=migrated_db)

    pipeline = IngestionPipeline(parser=parser, chunker=chunker, embedder=embedder, sink=store)
    uris = discover_sources(str(tmp_path))
    result = await pipeline.ingest(uris)
    assert result.failures == []
    assert result.n_documents == 2

    # Sparse search on a keyword present in the termination doc.
    hits = await store.sparse_search("thirty days notice", k=3)
    assert hits
    assert any("thirty" in h.chunk.text for h in hits)

    # Dense search: embed the query with the same fake backend, expect a match.
    qvec = (await embedder.embed(["Either party may terminate on thirty days notice."]))[0]
    dense = await store.dense_search(qvec, k=3)
    assert dense
    assert any("terminate" in h.chunk.text.lower() for h in dense)

    # Idempotence: rerunning ingest keeps the total chunk count stable.
    before = await store.count_chunks()
    result2 = await pipeline.ingest(uris)
    assert result2.failures == []
    after = await store.count_chunks()
    assert before == after
