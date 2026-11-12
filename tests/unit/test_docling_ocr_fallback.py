"""A PDF whose text layer is broken is read through OCR, or fails loudly.

Some official PDFs (cantonal CCT annexes) embed fonts whose ToUnicode map is wrong:
the page renders correctly but the extracted text is shifted letters, and digits come
out as control characters or U+FFFD. Docling OCRs such pages when asked to; these
tests pin that we ask, and that a page still broken after conversion fails the
document instead of being indexed with every figure gone. The slow lane
(``test_docling_parser.py``) runs the real OCR.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from verity.ingestion.parsers.docling_parser import (
    DoclingParser,
    UnreadableTextLayerError,
    _lazy_converter,
    broken_pages,
)
from verity.ingestion.structure import Section

_CLEAN = "Le salaire minimum d'un jardinier CFC est de CHF 26.05 de l'heure (art. 8.1)."
# What the broken annexe of the Vaud landscapers' CCT yields for one line.
_SHIFTED = "-B\x01QSÊTFOUF\x01DPOWFOUJPO\x01���\x01ThBQQMJRVF"


def _sections(*pages: tuple[str, int | None]) -> tuple[str, tuple[Section, ...]]:
    text, sections = "", []
    for page_text, page in pages:
        if text:
            text += "\n\n"
        start = len(text)
        text += page_text
        sections.append(Section(char_start=start, char_end=len(text), page=page))
    return text, tuple(sections)


def test_the_default_converter_ocrs_broken_pages() -> None:
    from docling.datamodel.base_models import InputFormat

    options = _lazy_converter().format_to_options[InputFormat.PDF].pipeline_options
    assert options.do_ocr
    assert options.ocr_options.replace_broken_text_layer


def test_clean_pages_are_not_broken() -> None:
    assert broken_pages(*_sections((_CLEAN, 1), (_CLEAN, 2))) == []


def test_only_the_broken_page_is_reported() -> None:
    """The real CCT has a clean first page (the cantonal decree) and a broken annexe."""
    pages = [(_CLEAN * 5, n) for n in range(1, 10)] + [(_SHIFTED, 10)]
    assert broken_pages(*_sections(*pages)) == [10]


def test_empty_text_has_no_broken_page() -> None:
    assert broken_pages("", ()) == []


def _converted(text: str) -> SimpleNamespace:
    item = SimpleNamespace(text=text, label="text", prov=[SimpleNamespace(page_no=1)])
    return SimpleNamespace(document=SimpleNamespace(iterate_items=lambda: iter([(item, 0)])))


class _Converter:
    def __init__(self, text: str) -> None:
        self._text = text

    def convert(self, _path: str) -> Any:
        return _converted(self._text)


@pytest.mark.asyncio
async def test_a_clean_pdf_is_parsed() -> None:
    parser = DoclingParser(converter=_Converter(_CLEAN))  # type: ignore[arg-type]
    document = await parser.parse("upload://cct.pdf", raw=b"%PDF")
    assert document.text == _CLEAN


@pytest.mark.asyncio
async def test_a_pdf_still_broken_after_conversion_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No OCR engine, or an OCR that read nothing better: indexing the garbage would
    make every figure of the document unanswerable while ingestion reports success."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    parser = DoclingParser(converter=_Converter(_SHIFTED))  # type: ignore[arg-type]
    with pytest.raises(UnreadableTextLayerError, match=r"page\(s\) \[1\]"):
        await parser.parse("upload://cct.pdf", raw=b"%PDF")
    assert not os.listdir(tmp_path)


@pytest.mark.asyncio
async def test_a_broken_docx_is_kept_as_is() -> None:
    """A DOCX has no rendered page to OCR; its text is what the file stores."""
    parser = DoclingParser(converter=_Converter(_SHIFTED))  # type: ignore[arg-type]
    document = await parser.parse("upload://memo.docx", raw=b"PK")
    assert document.text == _SHIFTED
