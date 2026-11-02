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

from tests.minimal_pdf import pdf_bytes
from verity.ingestion.parsers.docling_parser import DoclingParser
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


@pytest.mark.asyncio
async def test_corrupt_pdf_raises_and_leaves_no_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    with pytest.raises(ConversionError):
        await DoclingParser().parse("upload://bad.pdf", raw=b"%PDF-1.4 garbage")
    assert not os.listdir(tmp_path)
