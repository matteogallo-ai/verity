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
