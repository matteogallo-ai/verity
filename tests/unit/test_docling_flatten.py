"""DoclingParser's structure flattening and source handling, with a fake converter.

The real conversions (PDF, DOCX, corrupt PDF) are in ``test_docling_parser.py`` in the
slow lane; these cases need no model and run in the fast lane.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from verity.ingestion.parsers.docling_parser import DoclingParser, _flatten
from verity.types import SourceType


def _item(text: str, label: str = "text", level: int | None = None, page: int | None = None):  # type: ignore[no-untyped-def]
    prov = [SimpleNamespace(page_no=page)] if page is not None else []
    return SimpleNamespace(text=text, label=label, level=level, prov=prov)


def _doc(*items: Any) -> SimpleNamespace:
    return SimpleNamespace(iterate_items=lambda: [(i, 0) for i in items])


def test_headings_nest_by_level_and_reset_at_a_sibling() -> None:
    text, sections = _flatten(
        _doc(
            _item("Agreement", "title", level=1),
            _item("Terms", "section_header", level=2),
            _item("Body one.", page=3),
            _item("Fees", "section_header", level=2),
            _item("Body two."),
        )
    )
    assert text == "Agreement\n\nTerms\n\nBody one.\n\nFees\n\nBody two."
    assert sections[2].heading_path == ("Agreement", "Terms")
    assert sections[2].page == 3
    assert sections[4].heading_path == ("Agreement", "Fees")
    assert text[sections[4].char_start : sections[4].char_end] == "Body two."


def test_blank_items_are_skipped_and_a_heading_without_level_is_level_one() -> None:
    text, sections = _flatten(_doc(_item("   "), _item("Heading", "header"), _item("x")))
    assert text == "Heading\n\nx"
    assert sections[1].heading_path == ("Heading",)


def test_an_empty_document_has_one_empty_section() -> None:
    text, sections = _flatten(_doc())
    assert text == ""
    assert len(sections) == 1
    assert (sections[0].char_start, sections[0].char_end) == (0, 0)


def test_a_document_without_items_falls_back_to_its_plain_text() -> None:
    text, sections = _flatten(SimpleNamespace(export_to_text=lambda: "whole text"))
    assert text == "whole text"
    assert (sections[0].char_start, sections[0].char_end) == (0, 10)


class _Converter:
    def __init__(self, document: Any) -> None:
        self.document = document
        self.paths: list[str] = []

    def convert(self, path: str) -> SimpleNamespace:
        self.paths.append(path)
        return SimpleNamespace(document=self.document)


@pytest.mark.asyncio
async def test_a_local_file_is_converted_in_place(tmp_path: Path) -> None:
    source = tmp_path / "contract.docx"
    source.write_bytes(b"PK\x03\x04")
    converter = _Converter(_doc(_item("Body without heading.")))

    doc = await DoclingParser(converter).parse(f"file://{source}")  # type: ignore[arg-type]

    assert converter.paths == [str(source)]
    assert source.exists()
    assert doc.source_type is SourceType.DOCX
    assert doc.title == "Body without heading."


@pytest.mark.asyncio
async def test_an_empty_conversion_is_titled_after_the_uri_not_the_temp_file() -> None:
    converter = _Converter(_doc())
    doc = await DoclingParser(converter).parse("upload://scan.pdf", raw=b"%PDF-1.4")  # type: ignore[arg-type]
    assert doc.text == ""
    assert doc.source_type is SourceType.PDF
    assert doc.title == "scan"
    removed = not await asyncio.to_thread(os.path.exists, converter.paths[0])
    assert removed, "temporary copy must be removed"


def test_supports_pdf_and_docx_only() -> None:
    parser = DoclingParser()
    assert parser.supports(SourceType.PDF) and parser.supports(SourceType.DOCX)
    assert not parser.supports(SourceType.MARKDOWN)


def test_one_docling_converter_serves_every_document(monkeypatch: pytest.MonkeyPatch) -> None:
    """A converter loads docling's models on first use; one per document reloaded them."""
    import docling.document_converter

    from verity.ingestion.parsers import docling_parser

    built: list[object] = []
    monkeypatch.setattr(
        docling.document_converter, "DocumentConverter", lambda: built.append(1) or object()
    )
    docling_parser._lazy_converter.cache_clear()
    try:
        assert docling_parser._lazy_converter() is docling_parser._lazy_converter()
        assert len(built) == 1
    finally:
        docling_parser._lazy_converter.cache_clear()
