"""StubLLMClient: deterministic, offline, tracks calls."""

from __future__ import annotations

import pytest

from verity.llm.clients import StubLLMClient
from verity.types import Message, Provider, Role


@pytest.mark.asyncio
async def test_stub_returns_responder_output() -> None:
    client = StubLLMClient(lambda ms: f"echo:{ms[-1].content}")
    completion = await client.complete([Message(role=Role.USER, content="hi")])
    assert completion.text == "echo:hi"
    assert completion.provider is Provider.LOCAL


@pytest.mark.asyncio
async def test_stub_tracks_all_calls() -> None:
    client = StubLLMClient(lambda _: "ok")
    await client.complete([Message(role=Role.USER, content="a")])
    await client.complete([Message(role=Role.USER, content="b")])
    assert len(client.calls) == 2
    assert client.calls[0][-1].content == "a"
    assert client.calls[1][-1].content == "b"


@pytest.mark.asyncio
async def test_stub_reports_positive_usage() -> None:
    client = StubLLMClient(lambda _: "some output")
    out = await client.complete([Message(role=Role.USER, content="a longer prompt")])
    assert out.usage.input_tokens > 0
    assert out.usage.output_tokens > 0
    assert out.usage.llm_calls == 1
