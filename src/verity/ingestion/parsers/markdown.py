"""Markdown parser: preserves heading hierarchy in structural metadata.

The document text is the original markdown, verbatim with normalised newlines. Chunk
offsets therefore index the *raw markdown* — which is what we want, so citations point
at the source-of-truth file, not a lossy rendering.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

from verity.ingestion.ids import document_id_for_uri
from verity.ingestion.parsers.text import _normalize_newlines, _read_bytes
from verity.ingestion.structure import DocumentStructure, Section
from verity.types import Document, SourceType

_ATX_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", re.MULTILINE)


def _sections_from_headings(text: str) -> tuple[Section, ...]:
    """Split ``text`` into sections at ATX headings, keeping a materialised heading path.

    The section preceding the first heading (a preamble) is emitted with an empty path.
    Each section starts at the beginning of its heading line so that the heading text is
    included in the chunk that follows it — which is what a reader expects to see.
    """
    matches = list(_ATX_HEADING.finditer(text))
    if not matches:
        return (Section(char_start=0, char_end=len(text)),)

    sections: list[Section] = []
    heading_stack: list[tuple[int, str]] = []  # (level, heading_text)

    if matches[0].start() > 0:
        sections.append(Section(char_start=0, char_end=matches[0].start()))

    for i, match in enumerate(matches):
        level = len(match.group(1))
        heading = match.group(2).strip()
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, heading))

        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        path = tuple(h for _, h in heading_stack)
        sections.append(Section(char_start=start, char_end=end, heading_path=path))

    return tuple(sections)


class MarkdownParser:
    """Parses Markdown files, preserving heading hierarchy in metadata."""

    def supports(self, source_type: SourceType) -> bool:
        return source_type is SourceType.MARKDOWN

    async def parse(self, uri: str, raw: bytes | None = None) -> Document:
        data = _read_bytes(uri, raw)
        text = _normalize_newlines(data.decode("utf-8", errors="replace"))
        structure = DocumentStructure(sections=_sections_from_headings(text))
        title = _first_heading(text) or Path(urlparse(uri).path or uri).stem or None
        return Document(
            id=document_id_for_uri(uri),
            source_type=SourceType.MARKDOWN,
            uri=uri,
            title=title,
            text=text,
            metadata=structure.to_metadata(),
        )


def _first_heading(text: str) -> str | None:
    m = _ATX_HEADING.search(text)
    return m.group(2).strip() if m else None


__all__ = ["MarkdownParser"]
