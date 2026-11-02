"""Spending cap on one answer: checked before every LLM call, counted from real usage.

All clients are stubs; no paid model is called.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr
from typer.testing import CliRunner

from tests.unit.test_agent import _seed_retriever
from verity.agent.agent import create_default_agent
from verity.agent.scripted_stub import scripted_responder
from verity.config import ModelPrice, Settings
from verity.llm.budget import (
    MESSAGE_OVERHEAD_TOKENS,
    SpendCapExceededError,
    SpendCappedClient,
    worst_case_cost,
)
from verity.llm.clients import OpenAIClient, StubLLMClient
from verity.llm.routing import MultiProviderRoutingClient
from verity.types import Completion, ErrorKind, Message, Role

_QUESTION = "What is the notice period for terminating the master services agreement?"
_CENT = 0.01


class _CentPerCallStub(StubLLMClient):
    """Scripted stub whose every call is priced, estimated and billed at one cent."""

    def __init__(self) -> None:
        super().__init__(scripted_responder(), model="cent")
        self.max_tokens_seen: list[int] = []

    def cost_usd(self, *, input_tokens: int, output_tokens: int) -> float:
        return _CENT

    async def complete(self, messages: list[Message], **kwargs: Any) -> Completion:
        self.max_tokens_seen.append(kwargs["max_tokens"])
        base = await super().complete(messages, **kwargs)
        return base.model_copy(update={"usage": base.usage.model_copy(update={"cost_usd": _CENT})})


def _priced(model: str, input_usd: float, output_usd: float) -> OpenAIClient:
    settings = Settings(
        model_prices={
            model: ModelPrice(input_usd_per_mtok=input_usd, output_usd_per_mtok=output_usd)
        }
    )
    return OpenAIClient(api_key=SecretStr("k"), model=model, settings=settings)


def test_estimate_bounds_input_by_bytes_and_output_by_max_tokens_at_the_dearest_price() -> None:
    cheap, dear = _priced("cheap", 1.0, 2.0), _priced("dear", 3.0, 15.0)
    messages = [Message(role=Role.SYSTEM, content="sys"), Message(role=Role.USER, content="héllo")]
    n_bytes = len(b"sys") + len("héllo".encode())

    estimate = worst_case_cost([cheap, dear], messages, max_tokens=256)

    assert estimate == pytest.approx(
        dear.cost_usd(input_tokens=n_bytes + 2 * MESSAGE_OVERHEAD_TOKENS, output_tokens=256)
    )


def test_an_unpriced_stub_is_estimated_at_zero() -> None:
    stub = StubLLMClient(lambda _m: "x")
    assert worst_case_cost([stub], [Message(role=Role.USER, content="q")], max_tokens=10) == 0.0


@pytest.mark.asyncio
async def test_a_call_that_could_cross_the_cap_is_never_sent() -> None:
    stub = _CentPerCallStub()
    capped = SpendCappedClient(MultiProviderRoutingClient([stub]), cap_usd=0.025)
    messages = [Message(role=Role.USER, content="q")]

    await capped.complete(messages, max_tokens=8)
    await capped.complete(messages, max_tokens=8)
    with pytest.raises(SpendCapExceededError, match=r"\$0\.0200 of the \$0\.0250 cap"):
        await capped.complete(messages, max_tokens=8)

    assert len(stub.calls) == 2
    assert capped.spent.llm_calls == 2
    assert capped.spent.cost_usd == pytest.approx(0.02)


@pytest.mark.asyncio
async def test_the_agent_stops_with_a_typed_error_and_reports_what_it_spent() -> None:
    capped = SpendCappedClient(MultiProviderRoutingClient([_CentPerCallStub()]), cap_usd=0.025)
    agent = create_default_agent(retriever=await _seed_retriever(), client=capped)

    answer = await agent.answer(_QUESTION)

    assert answer.confidence is None
    assert answer.error is not None
    assert answer.error.kind is ErrorKind.BUDGET_EXCEEDED
    assert answer.usage.llm_calls == 2
    assert answer.usage.cost_usd == pytest.approx(0.02)


@pytest.mark.asyncio
async def test_an_answer_within_the_cap_is_unaffected() -> None:
    capped = SpendCappedClient(MultiProviderRoutingClient([_CentPerCallStub()]), cap_usd=0.03)
    agent = create_default_agent(retriever=await _seed_retriever(), client=capped)

    answer = await agent.answer(_QUESTION)

    assert answer.error is None
    assert answer.usage.llm_calls == 3


@pytest.mark.asyncio
async def test_every_agent_call_bounds_its_output_tokens() -> None:
    """The pre-call estimate is only an upper bound because ``max_tokens`` is one."""
    stub = _CentPerCallStub()
    agent = create_default_agent(
        retriever=await _seed_retriever(), client=MultiProviderRoutingClient([stub])
    )

    await agent.answer(_QUESTION)

    assert stub.max_tokens_seen == [256, 512, 256]


def test_ask_stops_at_the_budget_with_exit_4_and_prints_the_spend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import verity.agent.scripted_stub
    import verity.cli

    async def _retriever(**_: object) -> tuple[object, str]:
        return await _seed_retriever(), "memory"

    monkeypatch.setattr(verity.cli, "_build_retriever", _retriever)
    monkeypatch.setattr(
        verity.agent.scripted_stub,
        "build_stub_router",
        lambda: MultiProviderRoutingClient([_CentPerCallStub()]),
    )

    result = CliRunner().invoke(
        verity.cli.app,
        [
            "ask",
            _QUESTION,
            "--stub",
            "--corpus",
            str(Path("datasets/corpus")),
            "--budget-usd",
            "0.025",
        ],
        env={"COLUMNS": "200", "NO_COLOR": "1"},
    )

    assert result.exit_code == 4, result.output
    assert "budget_exceeded" in result.output
    assert "$0.0250 cap" in result.output
    assert "llm calls" in result.output
    assert "0.0200" in result.output
