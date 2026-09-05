"""Plain-text parser: no structure, one section spanning the whole document."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from verity.ingestion.ids import document_id_for_uri
from verity.ingestion.structure import DocumentStructure, Section
from verity.types import Document, SourceType


def _read_bytes(uri: str, raw: bytes | None) -> bytes:
    if raw is not None:
        return raw
    parsed = urlparse(uri)
    path = Path(parsed.path if parsed.scheme == "file" else uri)
    return path.read_bytes()


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


class TextParser:
    """Parses TXT files. The Document text is UTF-8 with normalised newlines."""

    def supports(self, source_type: SourceType) -> bool:
        return source_type is SourceType.TXT

    async def parse(self, uri: str, raw: bytes | None = None) -> Document:
        data = _read_bytes(uri, raw)
        text = _normalize_newlines(data.decode("utf-8", errors="replace"))
        structure = DocumentStructure(sections=(Section(char_start=0, char_end=len(text)),))
        return Document(
            id=document_id_for_uri(uri),
            source_type=SourceType.TXT,
            uri=uri,
            title=Path(urlparse(uri).path or uri).stem or None,
            text=text,
            metadata=structure.to_metadata(),
        )


__all__ = ["TextParser"]
