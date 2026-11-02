"""The corpus cap holds under concurrent uploads."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.unit.api_app import FakeEmbeddingBackend, FakeRerankerBackend
from verity.api.state import AppRuntime, CorpusFullError


@pytest.mark.asyncio
async def test_concurrent_uploads_cannot_overshoot_the_cap(tmp_path: Path) -> None:
    with (
        patch(
            "verity.ingestion.embedder._default_backend_factory",
            return_value=FakeEmbeddingBackend(384),
        ),
        patch(
            "verity.retrieval.reranker._default_backend_factory",
            return_value=FakeRerankerBackend(),
        ),
    ):
        runtime = AppRuntime(mode="stub", corpus_dir=tmp_path / "none", store="memory")
        results = await asyncio.gather(
            *(
                runtime.ingest_bytes(
                    filename=f"n{i}.md",
                    content=f"# Note {i}\n\nText {i}.\n".encode(),
                    extension=".md",
                    max_documents=1,
                )
                for i in range(4)
            ),
            return_exceptions=True,
        )

    assert sum(not isinstance(r, BaseException) for r in results) == 1
    assert all(isinstance(r, CorpusFullError) for r in results if isinstance(r, BaseException))
    assert len(await runtime.sources()) == 1
