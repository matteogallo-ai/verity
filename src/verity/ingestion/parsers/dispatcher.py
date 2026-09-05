"""Route a parse request to the concrete parser that knows the source type.

Detection is done here (not inside each parser) so callers never have to guess: give
the dispatcher a path or a URL and it figures out the format from the URI shape.
"""

from __future__ import annotations

from collections.abc import Sequence
from urllib.parse import urlparse

from verity.ingestion.base import Parser
from verity.types import Document, SourceType

_EXTENSION_MAP: dict[str, SourceType] = {
    ".txt": SourceType.TXT,
    ".md": SourceType.MARKDOWN,
    ".markdown": SourceType.MARKDOWN,
    ".pdf": SourceType.PDF,
    ".docx": SourceType.DOCX,
}


class UnsupportedSourceError(ValueError):
    """Raised when the dispatcher cannot determine a parser for a URI."""


def detect_source_type(uri: str) -> SourceType:
    """Detect the source type from ``uri`` alone (extension for files, scheme for URLs)."""
    parsed = urlparse(uri)
    if parsed.scheme in {"http", "https"}:
        return SourceType.WEB
    path = parsed.path or uri
    lowered = path.lower()
    for ext, source_type in _EXTENSION_MAP.items():
        if lowered.endswith(ext):
            return source_type
    raise UnsupportedSourceError(f"Unable to detect source type for uri: {uri!r}")


class SourceTypeParser:
    """A :class:`verity.ingestion.base.Parser` that routes by :class:`SourceType`.

    The dispatcher never touches the bytes itself — it delegates to the first
    registered parser whose ``supports`` returns True for the detected type.
    """

    def __init__(self, parsers: Sequence[Parser]) -> None:
        self._parsers: tuple[Parser, ...] = tuple(parsers)

    def supports(self, source_type: SourceType) -> bool:
        return any(p.supports(source_type) for p in self._parsers)

    async def parse(self, uri: str, raw: bytes | None = None) -> Document:
        source_type = detect_source_type(uri)
        for parser in self._parsers:
            if parser.supports(source_type):
                return await parser.parse(uri, raw)
        raise UnsupportedSourceError(
            f"No parser registered for source type {source_type} (uri={uri!r})"
        )


__all__ = ["SourceTypeParser", "UnsupportedSourceError", "detect_source_type"]
