"""Integration: the API on its default store, pgvector, with the real local models.

The real embedder is used on purpose: the shipped corpus is upserted into the shared
database, and later readers (the CI retrieval gate) expect its rows to carry real
vectors. No document is uploaded, so the test leaves the corpus as it found it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from verity.api import create_app
from verity.config import Settings
from verity.db import apply_pending
from verity.retrieval import PgVectorStore

pytestmark = pytest.mark.integration

CORPUS = Path("datasets/corpus")


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    url = os.environ.get("VERITY_DATABASE_URL", "postgresql://verity:verity@localhost:5432/verity")
    apply_pending(url)
    os.environ.setdefault("VERITY_DATABASE_URL", url)
    settings = Settings(api_auth_disabled=True)
    assert settings.store == "pgvector", "pgvector must be the default store"
    app = create_app(
        runs_dir=tmp_path_factory.mktemp("runs"),
        corpus_dir=CORPUS,
        agent_mode="stub",
        cors_origins=(),
        settings=settings,
    )
    return TestClient(app)


def test_status_lists_the_corpus_held_in_postgres(client: TestClient) -> None:
    response = client.get("/status")
    assert response.status_code == 200
    corpus = response.json()["corpus"]
    shipped = {p.name for p in CORPUS.glob("*.md")}
    assert shipped <= {Path(uri).name for uri in corpus["sources"]}
    assert corpus["n_chunks"] >= corpus["n_documents"] >= len(shipped)


def test_runtime_store_is_pgvector(client: TestClient) -> None:
    client.get("/status")
    assert isinstance(client.app.state.runtime.store, PgVectorStore)  # type: ignore[attr-defined]


def test_ask_answers_from_postgres_with_citations(client: TestClient) -> None:
    response = client.post(
        "/ask", json={"question": "What is the notice period for terminating the agreement?"}
    )
    assert response.status_code == 200
    answer = response.json()["answer"]
    assert answer["confidence"]["refused"] is False, answer
    assert answer["citations"]
