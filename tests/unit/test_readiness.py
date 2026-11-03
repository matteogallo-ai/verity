"""Liveness and readiness: ``/health`` says the process answers, ``/ready`` says it can
serve ``/ask`` (store reachable, at least one LLM provider reachable)."""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import Response
from pydantic import SecretStr

from tests.unit.api_app import FakeEmbeddingBackend, FakeRerankerBackend, api_app
from verity.config import Settings, get_settings
from verity.llm.base import AuthenticationError, ProviderUnavailableError
from verity.retrieval.base import StoreUnavailableError

_KEY = "test-key-0123456789"
_SETTINGS = Settings(api_keys=(SecretStr(_KEY),))
_STORE_PING = "verity.retrieval.pgvector_store.PgVectorStore.ping"
_ANTHROPIC_PING = "verity.llm.clients.AnthropicClient.ping"
_OPENAI_PING = "verity.llm.clients.OpenAIClient.ping"


@pytest.fixture(autouse=True)
def _provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Live mode builds its router only from providers with a key; no ping goes out."""
    monkeypatch.setenv("VERITY_ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("VERITY_OPENAI_API_KEY", "sk-test")
    get_settings.cache_clear()


async def _ok(*_a: object) -> None:
    return None


async def _down_store(*_a: object) -> None:
    raise StoreUnavailableError("cannot connect to Postgres: password=hunter2 host=db")


async def _down_provider(*_a: object) -> None:
    raise ProviderUnavailableError("anthropic: connection refused, key sk-ant-secret")


async def _hang(*_a: object) -> None:
    await asyncio.sleep(30)


def test_health_stays_200_when_every_dependency_is_down(tmp_path: Path) -> None:
    with (
        patch(_STORE_PING, _down_store),
        patch(_ANTHROPIC_PING, _down_provider),
        patch(_OPENAI_PING, _down_provider),
        api_app(tmp_path, _SETTINGS, store="pgvector", agent_mode="live") as client,
    ):
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503


def test_ready_on_the_memory_store_with_the_stub_agent(tmp_path: Path) -> None:
    with api_app(tmp_path, _SETTINGS) as client:
        r = client.get("/ready")
    assert r.status_code == 200
    assert r.json() == {
        "status": "ready",
        "checks": {"store": "not used", "llm": "not used", "preload": "ok"},
    }


def _until_ready(client: TestClient) -> Response:
    deadline = time.monotonic() + 10
    while (r := client.get("/ready")).status_code != 200:
        assert time.monotonic() < deadline, r.json()
        time.sleep(0.02)
    return r


def test_ready_is_503_until_the_corpus_and_models_are_loaded(tmp_path: Path) -> None:
    """The first ``/ask`` after ``/ready`` says 200 must not pay for loading models."""
    release = threading.Event()

    def slow_embedder(_model: str) -> FakeEmbeddingBackend:
        release.wait(10)
        return FakeEmbeddingBackend(384)

    with api_app(
        tmp_path, _SETTINGS, embedder_factory=slow_embedder, wait_for_preload=False
    ) as client:
        r = client.get("/ready")
        assert r.status_code == 503
        assert r.json()["checks"]["preload"] == "loading"
        assert client.get("/health").status_code == 200
        release.set()
        assert _until_ready(client).json()["checks"]["preload"] == "ok"


def test_startup_loads_both_models_before_any_request(tmp_path: Path) -> None:
    embedder = MagicMock(side_effect=lambda _m: FakeEmbeddingBackend(384))
    reranker = MagicMock(side_effect=lambda _m: FakeRerankerBackend())
    with api_app(tmp_path, _SETTINGS, embedder_factory=embedder, reranker_factory=reranker):
        assert embedder.call_count == 1
        assert reranker.call_count == 1


def test_a_failed_preload_is_not_ready_and_is_retried(tmp_path: Path) -> None:
    attempts: list[str] = []

    def flaky_embedder(_model: str) -> FakeEmbeddingBackend:
        attempts.append("load")
        if len(attempts) == 1:
            raise OSError("model not in the cache")
        return FakeEmbeddingBackend(384)

    with api_app(tmp_path, _SETTINGS, embedder_factory=flaky_embedder) as client:
        # One corpus file failed to embed: the corpus is incomplete, so not ready.
        assert client.app.state.runtime.preload_state == "failed"  # type: ignore[attr-defined]
        r = client.get("/ready")
        assert r.status_code == 503
        assert "cache" not in r.text
        # That call started another attempt, which loads the model this time.
        _until_ready(client)
        status = client.get("/status", headers={"X-API-Key": _KEY})
        n_docs = status.json()["corpus"]["n_documents"]
    assert len(attempts) == 2
    assert n_docs == len([p for p in Path("datasets/corpus").iterdir() if p.is_file()])


def test_ready_is_503_when_postgres_is_unreachable(tmp_path: Path) -> None:
    with patch(_STORE_PING, _down_store), api_app(tmp_path, _SETTINGS, store="pgvector") as client:
        r = client.get("/ready")
    assert r.status_code == 503
    assert r.json()["status"] == "not ready"
    assert r.json()["checks"]["store"] == "unreachable"
    # The cause goes to the log, never to an unauthenticated client.
    assert "hunter2" not in r.text and "Postgres" not in r.text


def test_ready_is_200_when_postgres_answers(tmp_path: Path) -> None:
    with patch(_STORE_PING, _ok), api_app(tmp_path, _SETTINGS, store="pgvector") as client:
        r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["checks"]["store"] == "ok"


def test_ready_is_503_when_no_provider_answers(tmp_path: Path) -> None:
    with (
        patch(_ANTHROPIC_PING, _down_provider),
        patch(_OPENAI_PING, _down_provider),
        api_app(tmp_path, _SETTINGS, agent_mode="live") as client,
    ):
        r = client.get("/ready")
    assert r.status_code == 503
    assert r.json()["checks"]["llm"] == "unreachable"
    assert "sk-ant" not in r.text


def test_one_reachable_provider_is_enough_because_routing_falls_back(tmp_path: Path) -> None:
    with (
        patch(_ANTHROPIC_PING, _down_provider),
        patch(_OPENAI_PING, _ok),
        api_app(tmp_path, _SETTINGS, agent_mode="live") as client,
    ):
        r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["checks"]["llm"] == "ok"


def test_a_rejected_key_is_not_ready(tmp_path: Path) -> None:
    async def bad_key(*_a: object) -> None:
        raise AuthenticationError("anthropic: 401")

    with (
        patch(_ANTHROPIC_PING, bad_key),
        patch(_OPENAI_PING, bad_key),
        api_app(tmp_path, _SETTINGS, agent_mode="live") as client,
    ):
        r = client.get("/ready")
    assert r.status_code == 503
    assert r.json()["checks"]["llm"] == "unreachable"


def test_a_hanging_dependency_is_cut_at_the_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("verity.api.readiness.CHECK_TIMEOUT_S", 0.05)
    with patch(_STORE_PING, _hang), api_app(tmp_path, _SETTINGS, store="pgvector") as client:
        r = client.get("/ready")
    assert r.status_code == 503
    assert r.json()["checks"]["store"] == "timeout"


def test_ready_needs_no_api_key(tmp_path: Path) -> None:
    with api_app(tmp_path, _SETTINGS) as client:
        assert client.get("/ready").status_code == 200


def test_a_rejected_first_key_is_not_ready_even_if_the_next_provider_answers(
    tmp_path: Path,
) -> None:
    """Routing falls back only on an unavailable provider; a rejected key stops it, so
    every ``/ask`` would fail although the second provider answers."""

    async def bad_key(*_a: object) -> None:
        raise AuthenticationError("anthropic: 401")

    with (
        patch(_ANTHROPIC_PING, bad_key),
        patch(_OPENAI_PING, _ok),
        api_app(tmp_path, _SETTINGS, agent_mode="live") as client,
    ):
        r = client.get("/ready")
    assert r.status_code == 503
    assert r.json()["checks"]["llm"] == "unreachable"


def test_back_to_back_probes_reach_the_dependencies_once(tmp_path: Path) -> None:
    """``/ready`` is public: without a short cache each call would open a Postgres
    connection and send a provider request."""
    calls: list[str] = []

    async def counted(*_a: object) -> None:
        calls.append("ping")

    with (
        patch(_STORE_PING, counted),
        patch(_ANTHROPIC_PING, counted),
        api_app(tmp_path, _SETTINGS, store="pgvector", agent_mode="live") as client,
    ):
        for _ in range(5):
            assert client.get("/ready").status_code == 200
    assert calls == ["ping", "ping"]  # one store ping, one provider ping


def test_a_failing_preload_is_retried_once_per_cache_period(tmp_path: Path) -> None:
    """Each attempt re-ingests the whole corpus; a caller looping on the public
    ``/ready`` must not turn that into continuous re-embedding."""
    loads: list[str] = []

    def broken_embedder(_model: str) -> FakeEmbeddingBackend:
        loads.append("load")
        raise OSError("model not in the cache")

    def settle(client: TestClient) -> None:
        deadline = time.monotonic() + 10
        while client.app.state.runtime.preload_state == "loading":  # type: ignore[attr-defined]
            assert time.monotonic() < deadline
            time.sleep(0.01)

    with api_app(tmp_path, _SETTINGS, embedder_factory=broken_embedder) as client:
        assert client.get("/ready").status_code == 503  # starts the one retry of this period
        settle(client)
        after_one_retry = len(loads)
        for _ in range(10):
            assert client.get("/ready").status_code == 503
            settle(client)
    assert len(loads) == after_one_retry


def test_requests_do_not_re_ingest_a_corpus_that_failed(tmp_path: Path) -> None:
    """If each ``/status`` call re-ingested a failing corpus, any key holder could keep
    the process embedding. Only the bounded readiness retry starts an attempt."""
    loads: list[str] = []

    def broken_embedder(_model: str) -> FakeEmbeddingBackend:
        loads.append("load")
        raise OSError("model not in the cache")

    with api_app(tmp_path, _SETTINGS, embedder_factory=broken_embedder) as client:
        after_startup = len(loads)
        for _ in range(3):
            assert client.get("/status", headers={"X-API-Key": _KEY}).status_code == 200
    assert len(loads) == after_startup
