"""Unit tests run without a database: the default store (pgvector) is swapped for the
in-memory one. A test that needs pgvector behaviour asks for it explicitly; the
integration lane (``tests/integration``) uses the real default."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from verity.config import get_settings


@pytest.fixture(autouse=True)
def _memory_store(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("VERITY_STORE", "memory")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True, scope="session")
def _parse_corpus_files_once() -> Iterator[None]:
    """Most unit tests seed a retriever from ``datasets/corpus``, whose PDF and DOCX
    documents go through docling (about 1-2 s per seeding). Parsing a file on disk is
    deterministic, so each one is parsed once per session; uploads (``raw`` bytes)
    and the docling parser tests that pass their own bytes are never cached."""
    from verity.ingestion.parsers.docling_parser import DoclingParser
    from verity.types import Document

    original = DoclingParser.parse
    cache: dict[str, Document] = {}

    async def parse(self: DoclingParser, uri: str, raw: bytes | None = None) -> Document:
        if raw is not None or self._converter is not None:
            return await original(self, uri, raw)
        if uri not in cache:
            cache[uri] = await original(self, uri, raw)
        return cache[uri]

    DoclingParser.parse = parse  # type: ignore[method-assign]
    yield
    DoclingParser.parse = original  # type: ignore[method-assign]
