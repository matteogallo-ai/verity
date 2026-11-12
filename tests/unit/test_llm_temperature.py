"""VERITY_LLM_TEMPERATURE: an opt-in sampling temperature for every model call (ADR 0009).

API version 2026-10-21, which the pinned anthropic SDK sends, rejects ``temperature``
with a 400. A call that sets one is sent on version 2023-06-01, which Claude Sonnet 4.6
still serves with a temperature; a call that sets none is the call ADR 0007 describes,
byte for byte. The real request is checked here through the SDK's own HTTP layer, on a
mock transport: no network, no key.
"""

from __future__ import annotations

import json
from typing import Any

import httpx2
import pytest
from pydantic import SecretStr, ValidationError

from verity.config import Settings
from verity.llm.clients import AnthropicClient, OpenAIClient
from verity.types import Message, Role

_MESSAGES = [Message(role=Role.SYSTEM, content="sys"), Message(role=Role.USER, content="hi")]
_REPLY = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-6",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 3, "output_tokens": 1},
}


def _sent_request(settings: Settings) -> httpx2.Request:
    import anthropic

    seen: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json=_REPLY)

    client = AnthropicClient(api_key=SecretStr("k"), settings=settings)
    client._client = anthropic.AsyncAnthropic(
        api_key="k", http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handle))
    )
    import asyncio

    completion = asyncio.run(client.complete(_MESSAGES, max_tokens=8))
    assert completion.text == "ok"
    (request,) = seen
    return request


def test_by_default_no_temperature_and_the_sdk_api_version_are_sent() -> None:
    from anthropic._constants import ANTHROPIC_API_VERSION

    request = _sent_request(Settings())
    assert request.headers["anthropic-version"] == ANTHROPIC_API_VERSION
    assert not {"temperature", "top_p", "top_k"} & set(json.loads(request.content))


def test_a_temperature_is_sent_on_the_api_version_that_accepts_it() -> None:
    request = _sent_request(Settings(llm_temperature=0.0))
    assert request.headers["anthropic-version"] == "2023-06-01"
    body = json.loads(request.content)
    assert body["temperature"] == 0.0
    assert not {"top_p", "top_k"} & set(body)


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_a_temperature_outside_zero_to_one_is_rejected(value: float) -> None:
    with pytest.raises(ValidationError):
        Settings(llm_temperature=value)


@pytest.mark.asyncio
async def test_openai_receives_the_same_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    """One sampling policy for every provider: the fallback samples as the primary."""
    from unittest.mock import AsyncMock, MagicMock

    sent: dict[str, Any] = {}

    async def create(**kwargs: Any) -> Any:
        sent.update(kwargs)
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="ok"))]
        resp.usage = MagicMock(prompt_tokens=1, completion_tokens=1)
        raw = MagicMock()
        raw.parse.return_value = resp
        raw.retries_taken = 0
        return raw

    fake = MagicMock()
    fake.chat.completions.with_raw_response.create = AsyncMock(side_effect=create)
    monkeypatch.setattr(OpenAIClient, "_get_client", lambda _self: fake)
    client = OpenAIClient(api_key=SecretStr("k"), settings=Settings(llm_temperature=0.0))
    await client.complete(_MESSAGES, max_tokens=8)
    assert sent["temperature"] == 0.0
