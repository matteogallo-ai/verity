"""Render the PDF and DOCX documents of the eval corpus from their Markdown sources.

``datasets/corpus_src/<name>.pdf.md`` becomes ``datasets/corpus/<name>.pdf`` and
``<name>.docx.md`` becomes ``<name>.docx``. The sources are the reviewable form; the
binaries are what the pipeline ingests, so PDF and DOCX parsing is part of every eval.

Supported Markdown: ``#``/``##``/``###`` headings, ``- `` bullets, paragraphs. The PDF
is written by hand (Helvetica, US Letter, no external dependency) and is
byte-for-byte reproducible; DOCX timestamps are pinned for the same reason.

    uv run python scripts/build_corpus_documents.py           # write
    uv run python scripts/build_corpus_documents.py --check   # exit 1 if a binary is stale
"""

from __future__ import annotations

import argparse
import io
import sys
import textwrap
import zipfile
from datetime import UTC, datetime
from pathlib import Path

SOURCES = Path("datasets/corpus_src")
CORPUS = Path("datasets/corpus")

_PAGE_W, _PAGE_H, _MARGIN = 612, 792, 72
# (font resource, size, leading, wrap width in characters) per block kind.
_STYLE = {
    "h1": ("F2", 16, 22, 55),
    "h2": ("F2", 13, 18, 70),
    "h3": ("F2", 11, 16, 85),
    "p": ("F1", 11, 15, 88),
    "li": ("F1", 11, 15, 84),
}
_PINNED = datetime(2025, 1, 1, tzinfo=UTC)


def _blocks(markdown: str) -> list[tuple[str, str]]:
    """``(kind, text)`` pairs; consecutive non-blank lines form one paragraph."""
    blocks: list[tuple[str, str]] = []
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            blocks.append(("p", " ".join(paragraph)))
            paragraph.clear()

    for raw in markdown.splitlines():
        line = raw.strip()
        if not line:
            flush()
        elif line.startswith("### "):
            flush()
            blocks.append(("h3", line[4:]))
        elif line.startswith("## "):
            flush()
            blocks.append(("h2", line[3:]))
        elif line.startswith("# "):
            flush()
            blocks.append(("h1", line[2:]))
        elif line.startswith("- "):
            flush()
            blocks.append(("li", line[2:]))
        else:
            paragraph.append(line)
    flush()
    return blocks


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def render_pdf(markdown: str) -> bytes:
    pages: list[list[str]] = [[]]
    y = _PAGE_H - _MARGIN
    for kind, text in _blocks(markdown):
        font, size, leading, width = _STYLE[kind]
        lines = textwrap.wrap(text, width=width) or [""]
        if kind == "li":
            lines = ["- " + lines[0]] + ["  " + rest for rest in lines[1:]]
        gap = 8 if kind.startswith("h") else 4
        for line in lines:
            if y - leading < _MARGIN:
                pages.append([])
                y = _PAGE_H - _MARGIN
            y -= leading
            pages[-1].append(f"BT /{font} {size} Tf {_MARGIN} {y} Td ({_pdf_escape(line)}) Tj ET")
        y -= gap

    n_pages = len(pages)
    # Objects: 1 catalog, 2 pages, 3 regular font, 4 bold font, then (page, contents) pairs.
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids ["
        + " ".join(f"{5 + 2 * i} 0 R" for i in range(n_pages))
        + f"] /Count {n_pages} >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>",
    ]
    for i, ops in enumerate(pages):
        stream = "\n".join(ops)
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {_PAGE_W} {_PAGE_H}] "
            f"/Contents {6 + 2 * i} 0 R /Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> >>"
        )
        objects.append(
            f"<< /Length {len(stream.encode('latin-1'))} >>\nstream\n{stream}\nendstream"
        )
    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode()
    out += f"startxref\n{xref}\n%%EOF\n".encode()
    return out


def render_docx(markdown: str) -> bytes:
    from docx import Document

    document = Document()
    document.core_properties.created = _PINNED
    document.core_properties.modified = _PINNED
    document.core_properties.last_modified_by = "verity"
    document.core_properties.author = "verity"
    for kind, text in _blocks(markdown):
        if kind.startswith("h"):
            document.add_heading(text, level=int(kind[1]))
        elif kind == "li":
            document.add_paragraph(text, style="List Bullet")
        else:
            document.add_paragraph(text)
    raw = io.BytesIO()
    document.save(raw)
    # Rewrite the archive with fixed entry timestamps so the bytes are reproducible.
    pinned = io.BytesIO()
    with zipfile.ZipFile(raw) as src, zipfile.ZipFile(pinned, "w", zipfile.ZIP_DEFLATED) as dst:
        for info in sorted(src.infolist(), key=lambda i: i.filename):
            entry = zipfile.ZipInfo(info.filename, date_time=(2025, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            dst.writestr(entry, src.read(info.filename))
    return pinned.getvalue()


def build() -> dict[Path, bytes]:
    outputs: dict[Path, bytes] = {}
    for source in sorted(SOURCES.glob("*.md")):
        stem, _, kind = source.name.removesuffix(".md").rpartition(".")
        if kind not in {"pdf", "docx"}:
            continue
        markdown = source.read_text(encoding="utf-8")
        markdown.encode("latin-1")  # the PDF fonts use WinAnsi; fail early on others
        render = render_pdf if kind == "pdf" else render_docx
        outputs[CORPUS / f"{stem}.{kind}"] = render(markdown)
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if a binary is stale")
    args = parser.parse_args(argv)
    stale = []
    for path, content in build().items():
        if not path.exists() or path.read_bytes() != content:
            stale.append(path)
            if not args.check:
                path.write_bytes(content)
    if args.check and stale:
        print("stale: " + ", ".join(str(p) for p in stale), file=sys.stderr)
        return 1
    print(("stale" if args.check else "wrote") + f" {len(stale)} document(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
