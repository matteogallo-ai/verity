"""Chunker: offset invariant, section attachment, deterministic ids.

The offset invariant is Verity's most load-bearing local invariant. If it breaks,
every citation stops pointing where the UI says it does. This test asserts it on
the full example corpus so any regression is caught immediately.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from verity.ingestion.chunker import StructureAwareChunker
from verity.ingestion.parsers import create_default_parser
from verity.ingestion.parsers.markdown import MarkdownParser
from verity.ingestion.pipeline import discover_sources

CORPUS = Path(__file__).resolve().parents[2] / "datasets" / "corpus"


def _load_corpus_documents():  # type: ignore[no-untyped-def]
    parser = create_default_parser()

    async def run():  # type: ignore[no-untyped-def]
        uris = discover_sources(str(CORPUS))
        return [await parser.parse(u) for u in uris]

    return asyncio.run(run())


def test_chunker_invariant_holds_on_corpus() -> None:
    docs = _load_corpus_documents()
    assert docs, "example corpus is empty — datasets/corpus/ must contain at least one doc"

    chunker = StructureAwareChunker()
    total_chunks = 0
    for doc in docs:
        chunks = chunker.chunk(doc)
        assert chunks, f"chunker produced no chunks for {doc.uri}"
        for chunk in chunks:
            substring = doc.text[chunk.char_start : chunk.char_end]
            assert substring == chunk.text, (
                f"offset invariant violated on {doc.uri} chunk#{chunk.ordinal}: "
                f"expected len={len(substring)} got len={len(chunk.text)}"
            )
        total_chunks += len(chunks)
    assert total_chunks > 0


@pytest.mark.asyncio
async def test_chunker_attaches_section_heading(tmp_path: Path) -> None:
    md = (
        "# Doc\n\n"
        "intro\n\n"
        "## Section A\n\n"
        "content of section A paragraph one.\n\n"
        "content of section A paragraph two.\n\n"
        "## Section B\n\n"
        "content of section B.\n"
    )
    p = tmp_path / "doc.md"
    p.write_text(md, encoding="utf-8")
    doc = await MarkdownParser().parse(str(p))
    chunks = StructureAwareChunker(target_words=10, max_words=30, min_words=1).chunk(doc)
    assert chunks
    for chunk in chunks:
        assert doc.text[chunk.char_start : chunk.char_end] == chunk.text
    sections = [c.section for c in chunks]
    assert any(s and "Doc > Section A" in s for s in sections)
    assert any(s and "Doc > Section B" in s for s in sections)


@pytest.mark.asyncio
async def test_chunker_warns_on_oversized_paragraph(tmp_path: Path) -> None:
    import structlog.testing

    huge = " ".join(["word"] * 200)  # single paragraph, 200 words
    p = tmp_path / "big.md"
    p.write_text(f"# H\n\n{huge}\n", encoding="utf-8")
    doc = await MarkdownParser().parse(str(p))
    chunker = StructureAwareChunker(target_words=50, max_words=100, min_words=1)
    with structlog.testing.capture_logs() as events:
        chunks = chunker.chunk(doc)
    assert chunks
    assert any(
        e.get("event") == "chunk_over_budget" and e.get("log_level") == "warning" for e in events
    ), f"expected chunk_over_budget WARN, got events={events!r}"


def test_chunker_does_not_warn_on_corpus() -> None:
    import structlog.testing

    docs = _load_corpus_documents()
    chunker = StructureAwareChunker()
    with structlog.testing.capture_logs() as events:
        for doc in docs:
            chunker.chunk(doc)
    over_budget = [e for e in events if e.get("event") == "chunk_over_budget"]
    assert not over_budget, f"corpus chunks unexpectedly triggered warnings: {over_budget!r}"


@pytest.mark.asyncio
async def test_chunker_deterministic_ids(tmp_path: Path) -> None:
    p = tmp_path / "doc.md"
    p.write_text("# H\n\nalpha beta gamma.\n\nsecond paragraph.\n", encoding="utf-8")
    doc = await MarkdownParser().parse(str(p))
    chunker = StructureAwareChunker()
    ids_first = [c.id for c in chunker.chunk(doc)]
    ids_second = [c.id for c in chunker.chunk(doc)]
    assert ids_first == ids_second
    assert len(ids_first) == len(set(ids_first))
