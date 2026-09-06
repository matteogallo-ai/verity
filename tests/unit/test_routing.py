"""RoutingClient: ordered fallback, AuthenticationError short-circuit, exhaustion."""

from __future__ import annotations

import pytest

from verity.llm.base import AuthenticationError, ProviderUnavailableError
from verity.llm.clients import StubLLMClient
from verity.llm.routing import MultiProviderRoutingClient
from verity.types import Message, Provider, Role


def _stub(text: str = "ok", *, provider: Provider = Provider.LOCAL) -> StubLLMClient:
    return StubLLMClient(lambda _messages: text, provider=provider)


def _failing_stub(exc: BaseException, *, provider: Provider = Provider.LOCAL) -> StubLLMClient:
    def _raise(_: object) -> str:
        raise exc

    return StubLLMClient(_raise, provider=provider)


@pytest.mark.asyncio
async def test_first_provider_wins_when_healthy() -> None:
    primary = _stub("primary", provider=Provider.ANTHROPIC)
    secondary = _stub("secondary", provider=Provider.OPENAI)
    router = MultiProviderRoutingClient([primary, secondary])
    out = await router.complete([Message(role=Role.USER, content="hi")])
    assert out.text == "primary"
    assert out.provider is Provider.ANTHROPIC
    assert primary.calls and not secondary.calls


@pytest.mark.asyncio
async def test_falls_back_on_provider_unavailable() -> None:
    down = _failing_stub(ProviderUnavailableError("boom"), provider=Provider.ANTHROPIC)
    up = _stub("saved", provider=Provider.OPENAI)
    router = MultiProviderRoutingClient([down, up])
    out = await router.complete([Message(role=Role.USER, content="hi")])
    assert out.text == "saved"
    assert out.provider is Provider.OPENAI


@pytest.mark.asyncio
async def test_authentication_error_short_circuits() -> None:
    bad_key = _failing_stub(AuthenticationError("401"), provider=Provider.ANTHROPIC)
    fallback = _stub("should-not-be-called", provider=Provider.OPENAI)
    router = MultiProviderRoutingClient([bad_key, fallback])
    with pytest.raises(AuthenticationError):
        await router.complete([Message(role=Role.USER, content="hi")])
    # Critical: the fallback provider MUST NOT be tried after an auth error.
    assert not fallback.calls, "AuthenticationError must short-circuit routing"


@pytest.mark.asyncio
async def test_exhaustion_raises_last_transient() -> None:
    first = _failing_stub(ProviderUnavailableError("first-down"), provider=Provider.ANTHROPIC)
    second = _failing_stub(ProviderUnavailableError("second-down"), provider=Provider.OPENAI)
    router = MultiProviderRoutingClient([first, second])
    with pytest.raises(ProviderUnavailableError, match="second-down"):
        await router.complete([Message(role=Role.USER, content="hi")])


def test_routing_client_requires_at_least_one() -> None:
    with pytest.raises(ValueError):
        MultiProviderRoutingClient([])
