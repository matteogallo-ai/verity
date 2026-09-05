"""PDF + DOCX parsing via docling.

Docling gives us a rich structured DoclingDocument. We flatten it into a plain text
string (headings + paragraphs, one item per line-block) while tracking, for each item,
its span into that string and its heading path. This gives the chunker exactly what it
needs — semantic boundaries + preserved offsets — while keeping the Document contract
minimal (``text`` is a plain string; structure lives in metadata).

Docling's model uses ``NodeItem.self_ref`` (opaque strings) rather than typed classes we
can import cheaply, so we detect item kinds by ``label``/attribute presence, which keeps
this module resilient to minor docling API drift.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlparse

from verity.ingestion.ids import document_id_for_uri
from verity.ingestion.parsers.text import _read_bytes
from verity.ingestion.structure import DocumentStructure, Section
from verity.types import Document, SourceType

if TYPE_CHECKING:  # pragma: no cover — types only, avoids importing at module load
    from docling.document_converter import DocumentConverter

# Docling labels that indicate a heading (varies by version; we accept a superset).
_HEADING_LABELS = {"title", "section_header", "subtitle", "header"}


def _lazy_converter() -> DocumentConverter:
    """Import docling only when a PDF/DOCX is actually parsed (cold-start cost)."""
    from docling.document_converter import DocumentConverter

    return DocumentConverter()


def _flatten(docling_doc: Any) -> tuple[str, tuple[Section, ...]]:
    """Walk a DoclingDocument and produce (plain_text, sections).

    Each item becomes one line-block in ``plain_text``, separated by a single blank
    line. Sections record the span of each item and the heading path that was in
    effect when the item was emitted.
    """
    parts: list[str] = []
    sections: list[Section] = []
    heading_stack: list[tuple[int, str]] = []
    cursor = 0
    separator = "\n\n"

    def _item_text(item: Any) -> str:
        text = getattr(item, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
        return ""

    def _item_page(item: Any) -> int | None:
        prov = getattr(item, "prov", None)
        if not prov:
            return None
        first = prov[0] if isinstance(prov, list | tuple) and prov else None
        page = getattr(first, "page_no", None) or getattr(first, "page", None)
        return int(page) if isinstance(page, int) else None

    def _item_label(item: Any) -> str:
        label = getattr(item, "label", "") or ""
        return str(label).lower()

    def _item_level(item: Any) -> int:
        level = getattr(item, "level", None)
        if isinstance(level, int) and level > 0:
            return level
        return 1

    iter_items = getattr(docling_doc, "iterate_items", None)
    if iter_items is None:
        # Fallback: single blob from the whole document.
        blob = docling_doc.export_to_text() if hasattr(docling_doc, "export_to_text") else ""
        text = blob if isinstance(blob, str) else ""
        return text, (Section(char_start=0, char_end=len(text)),)

    for item, _ in iter_items():
        text = _item_text(item)
        if not text:
            continue

        label = _item_label(item)
        if label in _HEADING_LABELS:
            level = _item_level(item)
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, text))

        if parts:
            cursor += len(separator)
            parts.append(separator)
        parts.append(text)

        span_start = cursor
        cursor += len(text)
        sections.append(
            Section(
                char_start=span_start,
                char_end=cursor,
                heading_path=tuple(h for _, h in heading_stack),
                page=_item_page(item),
            )
        )

    plain = "".join(parts)
    if not sections:
        sections.append(Section(char_start=0, char_end=len(plain)))
    return plain, tuple(sections)


class DoclingParser:
    """Structure-aware PDF and DOCX parser powered by docling."""

    def __init__(self, converter: DocumentConverter | None = None) -> None:
        self._converter = converter

    def supports(self, source_type: SourceType) -> bool:
        return source_type in {SourceType.PDF, SourceType.DOCX}

    async def parse(self, uri: str, raw: bytes | None = None) -> Document:
        source_type = SourceType.PDF if uri.lower().endswith(".pdf") else SourceType.DOCX
        converter = self._converter or _lazy_converter()

        parsed_uri = urlparse(uri)
        source_path: Path
        cleanup: Path | None = None
        if raw is not None:
            suffix = ".pdf" if source_type is SourceType.PDF else ".docx"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(raw)
                source_path = Path(tmp.name)
                cleanup = source_path
        else:
            source_path = Path(parsed_uri.path if parsed_uri.scheme == "file" else uri)
            if not source_path.exists():
                raw = _read_bytes(uri, None)
                suffix = ".pdf" if source_type is SourceType.PDF else ".docx"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(raw)
                    source_path = Path(tmp.name)
                    cleanup = source_path

        try:
            result = converter.convert(str(source_path))
            docling_doc = getattr(result, "document", result)
            text, sections = _flatten(docling_doc)
        finally:
            if cleanup is not None:
                cleanup.unlink(missing_ok=True)

        structure = DocumentStructure(sections=sections)
        title = _title_from_sections(sections, text) or source_path.stem or None
        return Document(
            id=document_id_for_uri(uri),
            source_type=source_type,
            uri=uri,
            title=title,
            text=text,
            metadata=structure.to_metadata(),
        )


def _title_from_sections(sections: tuple[Section, ...], text: str) -> str | None:
    for section in sections:
        if section.heading_path:
            return section.heading_path[0]
    if text.strip():
        first_line = text.strip().splitlines()[0]
        return first_line[:120] if first_line else None
    return None


# Re-export cast so mypy is happy with the TYPE_CHECKING guard.
_ = cast
