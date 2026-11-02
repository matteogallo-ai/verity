"""Unit tests run without a database: the default store (pgvector) is swapped for the
in-memory one. A test that needs pgvector behaviour asks for it explicitly; the
integration lane (``tests/integration``) uses the real default."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from verity.config import get_settings


@pytest.fixture(autouse=True)
def _memory_store(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("VERITY_STORE", "memory")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True, scope="session")
def _corpus_pdf_and_docx_from_the_fixture() -> Iterator[None]:
    """Most unit tests seed a retriever from ``datasets/corpus``. Its PDF and DOCX
    documents would need docling's models, which a fresh runner does not have and
    cannot download (the lane blocks the network), so their parsed form comes from
    the committed fixture (see ``tests/docling_fixture.py``; checked against a live
    parse in the slow lane). Uploads (``raw`` bytes) and the docling tests that pass
    their own converter still run docling."""
    from tests import docling_fixture
    from verity.ingestion.parsers.docling_parser import DoclingParser
    from verity.types import Document

    original = DoclingParser.parse
    parsed = docling_fixture.load()
    # Seeding passes relative or absolute paths; both name the same file.
    corpus_names: dict[str, str] = {}
    for path in docling_fixture.corpus_files():
        rel = path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path
        corpus_names.update({str(path): path.name, str(rel): path.name})

    async def parse(self: DoclingParser, uri: str, raw: bytes | None = None) -> Document:
        name = corpus_names.get(uri)
        if raw is None and self._converter is None and name is not None:
            return docling_fixture.document(uri, parsed[name])
        return await original(self, uri, raw)

    DoclingParser.parse = parse  # type: ignore[method-assign]
    yield
    DoclingParser.parse = original  # type: ignore[method-assign]
