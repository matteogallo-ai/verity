"""Parser dispatch + native TXT/Markdown behaviour.

PDF/DOCX (docling) and Web (trafilatura + httpx) are covered by their own tests when
integration deps are available; here we lock the dispatcher's routing and the pure-
Python parsers' offset/structure behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from verity.ingestion.parsers import detect_source_type
from verity.ingestion.parsers.dispatcher import (
    SourceTypeParser,
    UnsupportedSourceError,
)
from verity.ingestion.parsers.markdown import MarkdownParser
from verity.ingestion.parsers.text import TextParser
from verity.ingestion.structure import DocumentStructure
from verity.types import SourceType


def test_detect_source_type_by_extension(tmp_path: Path) -> None:
    assert detect_source_type("/x/y/z.md") is SourceType.MARKDOWN
    assert detect_source_type("/x/y/z.markdown") is SourceType.MARKDOWN
    assert detect_source_type("/x/y/z.txt") is SourceType.TXT
    assert detect_source_type("/x/y/z.pdf") is SourceType.PDF
    assert detect_source_type("/x/y/z.docx") is SourceType.DOCX
    assert detect_source_type("https://example.com/page") is SourceType.WEB
    assert detect_source_type("http://example.com/page.html") is SourceType.WEB
    with pytest.raises(UnsupportedSourceError):
        detect_source_type("/x/y/z.unknown")


@pytest.mark.asyncio
async def test_text_parser_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "note.txt"
    p.write_text("hello\r\nworld\r\n", encoding="utf-8")
    parser = TextParser()
    doc = await parser.parse(str(p))
    assert doc.source_type is SourceType.TXT
    assert doc.text == "hello\nworld\n"
    structure = DocumentStructure.from_metadata(doc.metadata)
    assert len(structure.sections) == 1


@pytest.mark.asyncio
async def test_markdown_preserves_heading_hierarchy(tmp_path: Path) -> None:
    md = (
        "preamble\n\n"
        "# Title\n\n"
        "intro paragraph\n\n"
        "## Section 1\n\n"
        "body of section one\n\n"
        "## Section 2\n\n"
        "body of section two\n"
    )
    p = tmp_path / "doc.md"
    p.write_text(md, encoding="utf-8")
    parser = MarkdownParser()
    doc = await parser.parse(str(p))
    structure = DocumentStructure.from_metadata(doc.metadata)
    # preamble + 3 heading sections
    assert len(structure.sections) == 4

    heading_paths = [tuple(s.heading_path) for s in structure.sections]
    assert heading_paths[0] == ()
    assert heading_paths[1] == ("Title",)
    assert heading_paths[2] == ("Title", "Section 1")
    assert heading_paths[3] == ("Title", "Section 2")

    # Sections are contiguous and cover the whole document text.
    assert structure.sections[0].char_start == 0
    assert structure.sections[-1].char_end == len(doc.text)
    for a, b in zip(structure.sections[:-1], structure.sections[1:], strict=True):
        assert a.char_end == b.char_start


@pytest.mark.asyncio
async def test_dispatcher_routes_by_source_type(tmp_path: Path) -> None:
    p = tmp_path / "note.txt"
    p.write_text("x", encoding="utf-8")
    parser = SourceTypeParser(parsers=(TextParser(), MarkdownParser()))
    doc = await parser.parse(str(p))
    assert doc.source_type is SourceType.TXT

    q = tmp_path / "note.md"
    q.write_text("# T\n\nbody", encoding="utf-8")
    doc2 = await parser.parse(str(q))
    assert doc2.source_type is SourceType.MARKDOWN


def test_dispatcher_supports_reports_union() -> None:
    parser = SourceTypeParser(parsers=(TextParser(), MarkdownParser()))
    assert parser.supports(SourceType.TXT)
    assert parser.supports(SourceType.MARKDOWN)
    assert not parser.supports(SourceType.PDF)
