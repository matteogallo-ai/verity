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

from typing import TYPE_CHECKING

import structlog

from verity.llm.base import (
    AuthenticationError,
    LLMClient,
    LLMError,
    ProviderUnavailableError,
    RoutingClient,
)
from verity.types import Completion, Message, Provider

if TYPE_CHECKING:
    from verity.config import Settings

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
    ) -> Completion:
        last_transient: ProviderUnavailableError | None = None
        for client in self._clients:
            try:
                return await client.complete(messages, max_tokens=max_tokens)
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

    A provider with no key is left out: its SDK fails before any request with an error
    the routing loop does not fall back on, so keeping it would make a one-key
    deployment refuse every question. With no key at all, construction fails here
    rather than on the first question.
    """
    from verity.config import get_settings, key_is_set
    from verity.llm.clients import AnthropicClient, OpenAIClient
    from verity.types import Provider

    settings = get_settings()
    clients: list[LLMClient] = []
    for provider in settings.provider_order:
        if provider is Provider.ANTHROPIC and include_anthropic:
            if key_is_set(settings.anthropic_api_key):
                clients.append(AnthropicClient(api_key=settings.anthropic_api_key))
        elif provider is Provider.OPENAI and include_openai and key_is_set(settings.openai_api_key):
            clients.append(OpenAIClient(api_key=settings.openai_api_key))
    if not clients:
        raise LLMError(
            "No provider in settings.provider_order has an API key; set "
            "VERITY_ANTHROPIC_API_KEY or VERITY_OPENAI_API_KEY"
        )
    return MultiProviderRoutingClient(clients)


def create_judge_client(
    *, agent_client: object, settings: Settings | None = None
) -> MultiProviderRoutingClient:
    """The judge's client: one provider, no fallback.

    ``judge_provider="auto"`` takes another model family when its key is set (OpenAI),
    otherwise Claude with ``judge_anthropic_model``. A fallback would swap the judge
    mid-run without changing its label, so there is none. A judge with the agent's
    own provider and model would grade its own answers and is refused.
    """
    from verity.config import get_settings
    from verity.llm.clients import AnthropicClient, OpenAIClient

    # The agent's temperature is for its answers; the default judge model (Claude 4.7+)
    # rejects any temperature, and a judge has no fallback to absorb the 400 (ADR 0009).
    settings = (settings or get_settings()).model_copy(update={"llm_temperature": None})
    provider = settings.judge_provider
    if provider == "auto":
        provider = "openai" if settings.openai_api_key is not None else "anthropic"
    judge: LLMClient
    if provider == "openai":
        judge = OpenAIClient(
            api_key=settings.openai_api_key, model=settings.judge_openai_model, settings=settings
        )
    else:
        judge = AnthropicClient(
            api_key=settings.anthropic_api_key,
            model=settings.judge_anthropic_model,
            settings=settings,
        )
    agent_provider, agent_model = primary_client_identity(agent_client)
    if (judge.provider, judge.model) == (agent_provider, agent_model):
        raise ValueError(
            f"The judge must not be the agent's model ({judge.model}); set "
            "VERITY_JUDGE_PROVIDER or VERITY_JUDGE_ANTHROPIC_MODEL / VERITY_JUDGE_OPENAI_MODEL."
        )
    return MultiProviderRoutingClient([judge])


def primary_client_identity(client: object) -> tuple[Provider | None, str | None]:
    """``(provider, model)`` of the first client a routing client would try.

    Used to stamp provenance before any request has been served (the agent's
    fallback stamp, the judge's model label, the scorecard's agent model). Works
    on any object exposing a ``clients`` sequence; anything else, or a client
    without a typed ``provider`` / string ``model``, yields ``None`` for that field.
    """
    clients = getattr(client, "clients", None)
    if not clients:
        return None, None
    first = clients[0]
    provider = getattr(first, "provider", None)
    model = getattr(first, "model", None)
    return (
        provider if isinstance(provider, Provider) else None,
        model if isinstance(model, str) else None,
    )


__all__ = [
    "MultiProviderRoutingClient",
    "create_default_routing_client",
    "primary_client_identity",
]
