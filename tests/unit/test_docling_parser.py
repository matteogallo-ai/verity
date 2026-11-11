"""PDF and DOCX parsing through docling, on real files built in memory.

Slow lane: the first PDF conversion downloads docling's layout and OCR models
(about 500 MB) and later runs read them from the local cache.
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

import pytest
from docling.exceptions import ConversionError
from docx import Document as DocxDocument
from PIL import Image, ImageDraw, ImageFont

from tests.minimal_pdf import broken_text_layer_pdf_bytes, pdf_bytes
from verity.ingestion.parsers.docling_parser import (
    DoclingParser,
    UnreadableTextLayerError,
    broken_pages,
)
from verity.ingestion.structure import DocumentStructure
from verity.types import SourceType

pytestmark = pytest.mark.slow

_SENTENCE = "Either party may terminate on thirty days notice."


def _docx_bytes() -> bytes:
    document = DocxDocument()
    document.add_heading("Termination", level=1)
    document.add_paragraph(_SENTENCE)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_docx_upload_keeps_heading_and_text() -> None:
    document = await DoclingParser().parse("upload://abc.docx", raw=_docx_bytes())
    assert document.source_type is SourceType.DOCX
    assert document.text == f"Termination\n\n{_SENTENCE}"
    assert document.title == "Termination"


@pytest.mark.asyncio
async def test_pdf_upload_extracts_its_text() -> None:
    document = await DoclingParser().parse("upload://abc.pdf", raw=pdf_bytes(_SENTENCE))
    assert document.source_type is SourceType.PDF
    assert _SENTENCE in document.text


_SALARY_LINES = (
    "Article 8 - Salaires",
    "Jardinier titulaire du CFC: 26.05 par heure",
    "soit 4775.85 par mois sur 2200 heures",
)


def _broken_salary_pdf() -> bytes:
    """The salary lines as an image, over a text layer that extracts as garbage."""
    image = Image.new("L", (1700, 300), 255)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=48)
    for i, line in enumerate(_SALARY_LINES):
        draw.text((20, 20 + 90 * i), line, fill=0, font=font)
    jpeg = io.BytesIO()
    image.save(jpeg, "JPEG", quality=95)
    return broken_text_layer_pdf_bytes(jpeg.getvalue(), 1700, 300, " ".join(_SALARY_LINES))


@pytest.mark.asyncio
async def test_a_page_with_a_broken_text_layer_is_read_through_ocr() -> None:
    document = await DoclingParser().parse("upload://cct.pdf", raw=_broken_salary_pdf())
    for figure in ("26.05", "4775.85", "2200"):
        assert figure in document.text
    assert (
        broken_pages(document.text, DocumentStructure.from_metadata(document.metadata).sections)
        == []
    )


@pytest.mark.asyncio
async def test_without_ocr_replacement_the_broken_page_fails_the_document() -> None:
    """Docling's default converter keeps the broken layer next to the OCR of the image."""
    from docling.document_converter import DocumentConverter

    with pytest.raises(UnreadableTextLayerError):
        await DoclingParser(converter=DocumentConverter()).parse(
            "upload://cct.pdf", raw=_broken_salary_pdf()
        )


@pytest.mark.asyncio
async def test_corrupt_pdf_raises_and_leaves_no_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    with pytest.raises(ConversionError):
        await DoclingParser().parse("upload://bad.pdf", raw=b"%PDF-1.4 garbage")
    assert not os.listdir(tmp_path)
