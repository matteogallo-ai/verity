"""A real FastAPI app for API tests: stub agent, shipped corpus, fake model backends.

No model is loaded and nothing touches the network: the embedder and reranker get
deterministic token-hash backends. The fake embedder reports 384 dimensions to match
``embedding_dim``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient

from verity.api import create_app
from verity.config import Settings


class FakeEmbeddingBackend:
    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> list[list[float]]:
        rows: list[list[float]] = []
        for s in sentences:
            v = [0.0] * self._dim
            for tok in s.lower().split():
                v[hash(tok) % self._dim] += 1.0
            n = sum(x * x for x in v) ** 0.5 or 1.0
            rows.append([x / n for x in v])
        return rows

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim


class FakeRerankerBackend:
    def predict(
        self,
        sentences: Sequence[tuple[str, str]],
        *,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> list[float]:
        return [float(len(set(q.lower().split()) & set(p.lower().split()))) for q, p in sentences]


async def _preloaded(self: Any) -> None:
    self.preload_state = "ok"


@contextmanager
def api_app(
    tmp_path: Path,
    settings: Settings,
    *,
    store: str = "memory",
    agent_mode: str = "stub",
    embedder_factory: Callable[[str], Any] | None = None,
    reranker_factory: Callable[[str], Any] | None = None,
    wait_for_preload: bool = True,
    corpus_dir: Path = Path("datasets/corpus"),
) -> Iterator[TestClient]:
    """The app on the in-memory store unless ``store`` says otherwise: the fast lane
    has no database. A ``live`` agent builds its provider clients but calls none.

    The app's startup runs, so the corpus and models preload; the client is handed
    over once that is settled unless ``wait_for_preload`` is false. With Postgres
    the preload is skipped, as there is no database to preload into."""
    settings = settings.model_copy(update={"store": store})
    with (
        patch(
            "verity.ingestion.embedder._default_backend_factory",
            embedder_factory or (lambda _model: FakeEmbeddingBackend(384)),
        ),
        patch(
            "verity.retrieval.reranker._default_backend_factory",
            reranker_factory or (lambda _model: FakeRerankerBackend()),
        ),
        patch("verity.api.state.AppRuntime._preload", _preloaded)
        if store != "memory"
        else nullcontext(),
    ):
        app = create_app(
            runs_dir=tmp_path / "runs",
            corpus_dir=corpus_dir,
            agent_mode=agent_mode,
            cors_origins=(),
            settings=settings,
        )
        with TestClient(app, raise_server_exceptions=False) as client:
            deadline = time.monotonic() + 30
            while wait_for_preload and app.state.runtime.preload_state == "loading":
                assert time.monotonic() < deadline, "preload did not settle"
                time.sleep(0.01)
            yield client
