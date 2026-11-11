"""A one-page PDF written by hand, so neither the tests nor the Docker smoke test
(which runs on the host's bare ``python3``) need a PDF library.

``python3 tests/minimal_pdf.py "some text" > out.pdf``
"""

from __future__ import annotations

import sys

# Maps every printable byte to the code point 29 below it: "A" extracts as "$", the
# space and the digits as control characters, as in the annexe of the Vaud
# landscapers' CCT.
_SHIFTED_TO_UNICODE = (
    b"/CIDInit /ProcSet findresource begin 12 dict begin begincmap /CMapName /Shifted def "
    b"/CMapType 2 def 1 begincodespacerange <00> <FF> endcodespacerange "
    b"1 beginbfrange <20> <7E> <0003> endbfrange endcmap "
    b"CMapName currentdict /CMap defineresource pop end end"
)


def pdf_bytes(text: str) -> bytes:
    """A one-page PDF with one line of Helvetica text."""
    return _pdf(
        f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode(),
        resources=b"/Font << /F1 5 0 R >>",
        objects=[b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"],
    )


def broken_text_layer_pdf_bytes(jpeg: bytes, width: int, height: int, hidden: str) -> bytes:
    """A page that shows a grayscale ``jpeg`` and whose text layer is ``hidden``, drawn
    in white through a font with a shifted ToUnicode map.

    The reader sees only the image; text extraction yields only garbage. A shifted map
    on a visible font is not enough: docling renders a non-embedded font through that
    map too, so OCR would read the same garbage the text layer holds.
    """
    content = (
        f"q 540 0 0 {540 * height // width} 36 600 cm /Im1 Do Q "
        f"BT 1 g /F1 10 Tf 36 100 Td ({hidden}) Tj ET"
    ).encode()
    return _pdf(
        content,
        resources=b"/Font << /F1 5 0 R >> /XObject << /Im1 7 0 R >>",
        objects=[
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
            b"/Encoding /WinAnsiEncoding /ToUnicode 6 0 R >>",
            _stream(b"", _SHIFTED_TO_UNICODE),
            _stream(
                f"/Type /XObject /Subtype /Image /Width {width} /Height {height} "
                "/ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /DCTDecode ".encode(),
                jpeg,
            ),
        ],
    )


def _stream(dictionary: bytes, data: bytes) -> bytes:
    return b"<< " + dictionary + b"/Length %d >>\nstream\n" % len(data) + data + b"\nendstream"


def _pdf(content: bytes, *, resources: bytes, objects: list[bytes]) -> bytes:
    """Objects 1-4 are the catalog, page tree, page and content; ``objects`` follow from 5."""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << " + resources + b" >> >>",
        _stream(b"", content),
        *objects,
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode()
    out += f"startxref\n{xref}\n%%EOF\n".encode()
    return out


if __name__ == "__main__":
    sys.stdout.buffer.write(pdf_bytes(sys.argv[1]))
