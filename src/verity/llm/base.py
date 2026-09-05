"""LLM runtime contracts: a single-provider client and a routing client with fallback.

The routing behaviour is intentionally identical to the one proven in PromptLang's
RoutingClient: try providers in order, fall back on transient/availability errors, but
**short-circuit on authentication errors** — a bad key is a configuration bug, not a
reason to silently burn the fallback provider. That decision is recorded and not to be
revisited.

Implementations land in S3.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from verity.types import Completion, Message, Provider


class LLMError(Exception):
    """Base class for LLM runtime errors."""


class AuthenticationError(LLMError):
    """Invalid/missing credentials. Short-circuits the routing fallback — never retried
    on another provider, because a misconfigured key is not a transient failure."""


class ProviderUnavailableError(LLMError):
    """Transient failure (rate limit, timeout, 5xx). Eligible for fallback."""


@runtime_checkable
class LLMClient(Protocol):
    """A single concrete provider (Anthropic, OpenAI, or a local model server)."""

    @property
    def provider(self) -> Provider: ...

    @property
    def model(self) -> str: ...

    async def complete(
        self, messages: list[Message], *, max_tokens: int = 1024, temperature: float = 0.0
    ) -> Completion: ...


@runtime_checkable
class RoutingClient(Protocol):
    """Fronts an ordered set of :class:`LLMClient` with automatic fallback.

    Contract:
      * try clients in configured order;
      * on :class:`ProviderUnavailableError`, advance to the next client;
      * on :class:`AuthenticationError`, raise immediately (no fallback);
      * if all providers are exhausted, raise the last transient error.
    """

    async def complete(
        self, messages: list[Message], *, max_tokens: int = 1024, temperature: float = 0.0
    ) -> Completion: ...
