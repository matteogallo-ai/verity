"""Parser implementations and default factory.

The public surface here is a single function, :func:`create_default_parser`, that
returns a :class:`Parser` conforming to the ingestion protocol and knowing about
every supported :class:`SourceType`. Downstream code (CLI, pipeline, tests) depends
on the protocol, never on a concrete parser.
"""

from __future__ import annotations

from verity.ingestion.base import Parser
from verity.ingestion.parsers.dispatcher import (
    SourceTypeParser,
    detect_source_type,
)
from verity.ingestion.parsers.docling_parser import DoclingParser
from verity.ingestion.parsers.markdown import MarkdownParser
from verity.ingestion.parsers.text import TextParser
from verity.ingestion.parsers.web import WebParser


def create_default_parser() -> Parser:
    """Return the production parser: dispatcher over all built-in format handlers."""
    return SourceTypeParser(
        parsers=(
            TextParser(),
            MarkdownParser(),
            DoclingParser(),
            WebParser(),
        )
    )


__all__ = [
    "DoclingParser",
    "MarkdownParser",
    "SourceTypeParser",
    "TextParser",
    "WebParser",
    "create_default_parser",
    "detect_source_type",
]
