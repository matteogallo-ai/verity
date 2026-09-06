"""Ordered fallback routing across multiple LLM providers.

Contract (mirrors PromptLang's RoutingClient; recorded in ``verity.llm.base``):

- try clients in the order they were registered;
- on :class:`ProviderUnavailableError`, advance to the next client;
- on :class:`AuthenticationError`, RAISE IMMEDIATELY — a bad key is a config bug and
  must not silently burn the fallback provider (audit-trail intent, not a heuristic);
- when every provider has been exhausted, raise the LAST transient error observed.

The client list is a tuple of concrete :class:`LLMClient` implementations so the
routing behaviour is testable without a network — a stub returning a scripted
exception is a perfectly valid element.
"""

from __future__ import annotations

import structlog

from verity.llm.base import (
    AuthenticationError,
    LLMClient,
    LLMError,
    ProviderUnavailableError,
    RoutingClient,
)
from verity.types import Completion, Message

log = structlog.get_logger(__name__)


class MultiProviderRoutingClient(RoutingClient):
    """RoutingClient over an ordered list of ``LLMClient`` instances."""

    def __init__(self, clients: list[LLMClient]) -> None:
        if not clients:
            raise ValueError("MultiProviderRoutingClient requires at least one client")
        self._clients: tuple[LLMClient, ...] = tuple(clients)

    @property
    def clients(self) -> tuple[LLMClient, ...]:
        return self._clients

    async def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Completion:
        last_transient: ProviderUnavailableError | None = None
        for client in self._clients:
            try:
                return await client.complete(
                    messages, max_tokens=max_tokens, temperature=temperature
                )
            except AuthenticationError:
                # Short-circuit: never silently try the next provider.
                log.error(
                    "auth_error_short_circuit",
                    provider=client.provider.value,
                    model=client.model,
                )
                raise
            except ProviderUnavailableError as exc:
                log.warning(
                    "provider_unavailable_fallback",
                    provider=client.provider.value,
                    model=client.model,
                    error=str(exc),
                )
                last_transient = exc
                continue
        # Every provider raised a transient failure.
        assert last_transient is not None, "routing loop exited without a transient error"
        raise last_transient


def create_default_routing_client(
    *,
    include_anthropic: bool = True,
    include_openai: bool = True,
) -> MultiProviderRoutingClient:
    """Build a RoutingClient honouring ``settings.provider_order``.

    Providers with no key configured are still constructed — the SDK client fails at
    request time and the routing loop falls back to the next. This is intentional: a
    misconfiguration surfaces as a runtime error the first time an LLM call is made,
    not silently at process start.
    """
    from verity.config import get_settings
    from verity.llm.clients import AnthropicClient, OpenAIClient
    from verity.types import Provider

    settings = get_settings()
    clients: list[LLMClient] = []
    for provider in settings.provider_order:
        if provider is Provider.ANTHROPIC and include_anthropic:
            clients.append(AnthropicClient(api_key=settings.anthropic_api_key))
        elif provider is Provider.OPENAI and include_openai:
            clients.append(OpenAIClient(api_key=settings.openai_api_key))
    if not clients:
        raise LLMError("No providers configured; check settings.provider_order")
    return MultiProviderRoutingClient(clients)


__all__ = ["MultiProviderRoutingClient", "create_default_routing_client"]
