"""Liveness and readiness: ``/health`` says the process answers, ``/ready`` says it can
serve ``/ask`` (store reachable, at least one LLM provider reachable)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from tests.unit.api_app import api_app
from verity.config import Settings
from verity.llm.base import AuthenticationError, ProviderUnavailableError
from verity.retrieval.base import StoreUnavailableError

_KEY = "test-key-0123456789"
_SETTINGS = Settings(api_keys=(SecretStr(_KEY),))
_STORE_PING = "verity.retrieval.pgvector_store.PgVectorStore.ping"
_ANTHROPIC_PING = "verity.llm.clients.AnthropicClient.ping"
_OPENAI_PING = "verity.llm.clients.OpenAIClient.ping"


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
    assert r.json() == {"status": "ready", "checks": {"store": "not used", "llm": "not used"}}


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
