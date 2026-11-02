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


def _pdf_bytes(text: str) -> bytes:
    """A one-page PDF with one line of Helvetica text, written by hand so the test
    needs no PDF library."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode()
    out += f"startxref\n{xref}\n%%EOF\n".encode()
    return out


@pytest.mark.asyncio
async def test_docx_upload_keeps_heading_and_text() -> None:
    document = await DoclingParser().parse("upload://abc.docx", raw=_docx_bytes())
    assert document.source_type is SourceType.DOCX
    assert document.text == f"Termination\n\n{_SENTENCE}"
    assert document.title == "Termination"


@pytest.mark.asyncio
async def test_pdf_upload_extracts_its_text() -> None:
    document = await DoclingParser().parse("upload://abc.pdf", raw=_pdf_bytes(_SENTENCE))
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
