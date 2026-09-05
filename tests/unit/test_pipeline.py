"""Pipeline: dry-run over the example corpus without a DB or a real embedder."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from verity.ingestion import StructureAwareChunker, create_default_parser
from verity.ingestion.embedder import LocalEmbedder
from verity.ingestion.pipeline import (
    IngestionPipeline,
    discover_sources,
)
from verity.types import Chunk, EmbeddedChunk

CORPUS = Path(__file__).resolve().parents[2] / "datasets" / "corpus"


class _FakeBackend:
    def __init__(self, dim: int = 4) -> None:
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
        return [[0.1] * self._dim for _ in sentences]

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim


class _RecordingSink:
    def __init__(self) -> None:
        self.documents: list = []  # type: ignore[type-arg]
        self.chunks: list[EmbeddedChunk] = []

    async def upsert_documents(self, documents) -> None:  # type: ignore[no-untyped-def]
        self.documents.extend(documents)

    async def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        self.chunks.extend(chunks)


def test_discover_sources_finds_corpus_docs() -> None:
    uris = discover_sources(str(CORPUS))
    assert len(uris) >= 2
    assert all(u.endswith((".md", ".markdown", ".txt", ".pdf", ".docx")) for u in uris)


def test_discover_sources_rejects_missing() -> None:
    import pytest as _pytest

    with _pytest.raises(FileNotFoundError):
        discover_sources("/nonexistent/path/verity-does-not-exist")


@pytest.mark.asyncio
async def test_pipeline_ingests_corpus_with_fake_backend() -> None:
    parser = create_default_parser()
    chunker = StructureAwareChunker()
    embedder = LocalEmbedder(model="fake", expected_dim=4, backend=_FakeBackend(4))
    sink = _RecordingSink()

    pipeline = IngestionPipeline(parser=parser, chunker=chunker, embedder=embedder, sink=sink)
    uris = discover_sources(str(CORPUS))
    result = await pipeline.ingest(uris)

    assert result.failures == []
    assert result.n_documents == len(uris)
    assert result.n_chunks == len(sink.chunks)
    assert len(sink.documents) == len(uris)

    # Idempotence: rerunning produces the same chunk ids.
    sink2 = _RecordingSink()
    pipeline2 = IngestionPipeline(parser=parser, chunker=chunker, embedder=embedder, sink=sink2)
    result2 = await pipeline2.ingest(uris)
    ids_1 = sorted(str(c.chunk.id) for c in sink.chunks)
    ids_2 = sorted(str(c.chunk.id) for c in sink2.chunks)
    assert ids_1 == ids_2
    assert result2.n_documents == result.n_documents


@pytest.mark.asyncio
async def test_pipeline_survives_parse_failure(tmp_path: Path) -> None:
    class _AlwaysFailParser:
        def supports(self, source_type):  # type: ignore[no-untyped-def]
            return True

        async def parse(self, uri: str, raw: bytes | None = None):
            raise RuntimeError("boom")

    class _NoopChunker:
        def chunk(self, document):  # type: ignore[no-untyped-def]
            return []

    class _NoopEmbedder:
        @property
        def model(self) -> str:
            return "noop"

        @property
        def dim(self) -> int:
            return 0

        async def embed(self, texts):  # type: ignore[no-untyped-def]
            return []

        async def embed_chunks(self, chunks: list[Chunk]):  # type: ignore[no-untyped-def]
            return []

    pipeline = IngestionPipeline(
        parser=_AlwaysFailParser(),  # type: ignore[arg-type]
        chunker=_NoopChunker(),
        embedder=_NoopEmbedder(),
    )
    result = await pipeline.ingest(["a.txt", "b.txt", "c.txt"])
    assert result.n_documents == 0
    assert len(result.failures) == 3
    assert all("parse: boom" in f.error for f in result.failures)
