"""One structured log line per request: request ID, stage latencies, tokens, cost and
the decision (answered / refused / error)."""

from __future__ import annotations

from pathlib import Path

import pytest
import structlog
from pydantic import SecretStr

from tests.unit.api_app import api_app
from tests.unit.test_agent import _seed_retriever
from verity.agent.agent import create_default_agent
from verity.agent.scripted_stub import build_stub_router
from verity.config import Settings
from verity.llm.base import ProviderUnavailableError
from verity.llm.clients import StubLLMClient
from verity.llm.routing import MultiProviderRoutingClient
from verity.types import Message

_STAGES = {"agent.decompose", "agent.retrieve", "agent.synthesize", "agent.confidence"}


def _done(events: list[dict[str, object]]) -> dict[str, object]:
    lines = [e for e in events if e["event"] in {"answer_done", "answer_failed"}]
    assert len(lines) == 1, lines
    return lines[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("question", "decision"),
    [
        ("What is the notice period for terminating the master services agreement?", "answered"),
        ("What is the CEO's home address?", "refused"),
    ],
)
async def test_every_answer_logs_one_line_with_stages_usage_and_decision(
    question: str, decision: str
) -> None:
    agent = create_default_agent(retriever=await _seed_retriever(), client=build_stub_router())
    with structlog.testing.capture_logs() as events:
        answer = await agent.answer(question)

    line = _done(events)
    assert line["decision"] == decision
    assert set(line["stage_ms"]) == _STAGES  # type: ignore[arg-type]
    assert all(v >= 0 for v in line["stage_ms"].values())  # type: ignore[union-attr]
    assert answer.usage is not None
    assert line["input_tokens"] == answer.usage.input_tokens
    assert line["output_tokens"] == answer.usage.output_tokens
    assert line["cost_usd"] == answer.usage.cost_usd
    assert line["trace_id"] == str(answer.trace_id)


@pytest.mark.asyncio
async def test_an_outage_logs_decision_error_with_the_stages_reached() -> None:
    def _down(_messages: list[Message]) -> str:
        raise ProviderUnavailableError("down")

    agent = create_default_agent(
        retriever=await _seed_retriever(),
        client=MultiProviderRoutingClient([StubLLMClient(_down)]),
    )
    with structlog.testing.capture_logs() as events:
        await agent.answer("What is the notice period?")

    line = _done(events)
    assert line["decision"] == "error"
    assert line["error_kind"] == "provider_unavailable"
    assert set(line["stage_ms"]) == {"agent.decompose"}  # type: ignore[arg-type]


def test_the_api_log_line_carries_the_request_id(tmp_path: Path) -> None:
    settings = Settings(api_keys=(SecretStr("k" * 20),))
    with api_app(tmp_path, settings) as client:
        client.post("/ask", json={"question": "warm-up"}, headers={"X-API-Key": "k" * 20})
        with structlog.testing.capture_logs(
            processors=[structlog.contextvars.merge_contextvars]
        ) as events:
            response = client.post(
                "/ask",
                json={"question": "What is the CEO's home address?"},
                headers={"X-API-Key": "k" * 20, "X-Request-ID": "req-42"},
            )
    assert response.status_code == 200
    line = _done(events)
    assert line["request_id"] == "req-42"
    assert line["principal"] == "key-0"
