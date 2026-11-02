"""LLMClient implementations for Anthropic, OpenAI, and a deterministic stub.

The real clients wrap each vendor's async SDK behind the ``LLMClient`` protocol and
translate vendor-specific error shapes into the two Verity errors the routing client
branches on (``AuthenticationError`` short-circuits; ``ProviderUnavailableError``
triggers fallback). Any other exception propagates unchanged.

``StubLLMClient`` is what makes the eval harness and CI free: it takes a
``responder`` callable, deterministically maps a message list to a completion, and
never touches a network. Every agent test constructs the agent with a stub, so no
key is needed to prove the pipeline is wired correctly (not that the model behaves).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, NoReturn

from pydantic import SecretStr

from verity.config import Settings, get_settings
from verity.llm.base import AuthenticationError, LLMClient, LLMError, ProviderUnavailableError
from verity.types import Completion, Message, Provider, UsageStats

# --------------------------------------------------------------------------------------
# Stub
# --------------------------------------------------------------------------------------

Responder = Callable[[list[Message]], str]


class StubLLMClient(LLMClient):
    """Deterministic LLM stub. Never touches a network, never needs a key.

    Construct with a ``responder(messages) -> str``: it receives the exact message
    list the agent would send to a real provider and returns the assistant text.
    The stub records every call (``.calls``) so tests can assert on the prompts
    that were actually rendered.
    """

    def __init__(
        self,
        responder: Responder,
        *,
        model: str = "stub",
        provider: Provider = Provider.LOCAL,
    ) -> None:
        self._responder = responder
        self._model = model
        self._provider = provider
        self.calls: list[list[Message]] = []

    @property
    def provider(self) -> Provider:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    async def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
    ) -> Completion:
        started = time.perf_counter()
        self.calls.append(list(messages))
        text = self._responder(messages)
        elapsed_ms = (time.perf_counter() - started) * 1000
        input_chars = sum(len(m.content) for m in messages)
        return Completion(
            text=text,
            provider=self._provider,
            model=self._model,
            usage=UsageStats(
                input_tokens=max(1, input_chars // 4),
                output_tokens=max(1, len(text) // 4),
                llm_calls=1,
                cost_usd=0.0,
                latency_ms=round(elapsed_ms, 2),
            ),
        )


# --------------------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------------------


def _lazy_anthropic() -> Any:
    import anthropic

    return anthropic


class AnthropicClient(LLMClient):
    """Anthropic Messages API. Model + key resolved from config/env.

    No sampling parameter is sent: API version 2026-10-21 rejects ``temperature``,
    ``top_p`` and ``top_k`` (400), and Claude 4.7+ models accept only the defaults.
    Eval reproducibility rests on the recorded artefacts, not on sampling (ADR 0007).
    """

    def __init__(
        self,
        *,
        api_key: SecretStr | None = None,
        model: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._api_key = api_key
        self._model = model or self._settings.anthropic_model
        self._price = self._settings.price_for(self._model)
        self._client: Any = None

    @property
    def provider(self) -> Provider:
        return Provider.ANTHROPIC

    @property
    def model(self) -> str:
        return self._model

    def cost_usd(self, *, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self._price.input_usd_per_mtok
            + output_tokens * self._price.output_usd_per_mtok
        ) / 1_000_000

    def _get_client(self) -> Any:
        if self._client is None:
            anthropic = _lazy_anthropic()
            try:
                self._client = anthropic.AsyncAnthropic(
                    api_key=_unwrap(self._api_key),
                    timeout=self._settings.llm_timeout_s,
                    max_retries=self._settings.llm_max_retries,
                )
            # Any constructor failure (missing key, bad option) means this provider
            # cannot serve; typed as unavailable so the router tries the next one.
            except Exception as exc:
                raise ProviderUnavailableError(f"anthropic init failed: {exc}") from exc
        return self._client

    async def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
    ) -> Completion:
        client = self._get_client()
        started = time.perf_counter()
        system, chat = _split_system(messages)
        try:
            resp = await client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": m.role.value, "content": m.content} for m in chat],
            )
        # Every vendor failure is translated to a typed Verity error and re-raised.
        except Exception as exc:
            _raise_translated(exc, sdk=_lazy_anthropic(), provider="anthropic")
        elapsed_ms = (time.perf_counter() - started) * 1000
        text = _anthropic_text(resp)
        usage = getattr(resp, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        return Completion(
            text=text,
            provider=self.provider,
            model=self._model,
            usage=UsageStats(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                llm_calls=1,
                cost_usd=self.cost_usd(input_tokens=input_tokens, output_tokens=output_tokens),
                latency_ms=round(elapsed_ms, 2),
            ),
        )


def _anthropic_text(response: Any) -> str:
    blocks = getattr(response, "content", None) or []
    parts: list[str] = []
    for block in blocks:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


# --------------------------------------------------------------------------------------
# OpenAI
# --------------------------------------------------------------------------------------


def _lazy_openai() -> Any:
    import openai

    return openai


class OpenAIClient(LLMClient):
    """OpenAI Chat Completions. Same error-translation contract as ``AnthropicClient``."""

    def __init__(
        self,
        *,
        api_key: SecretStr | None = None,
        model: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._api_key = api_key
        self._model = model or self._settings.openai_model
        self._price = self._settings.price_for(self._model)
        self._client: Any = None

    @property
    def provider(self) -> Provider:
        return Provider.OPENAI

    @property
    def model(self) -> str:
        return self._model

    def cost_usd(self, *, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self._price.input_usd_per_mtok
            + output_tokens * self._price.output_usd_per_mtok
        ) / 1_000_000

    def _get_client(self) -> Any:
        if self._client is None:
            openai = _lazy_openai()
            try:
                self._client = openai.AsyncOpenAI(
                    api_key=_unwrap(self._api_key),
                    timeout=self._settings.llm_timeout_s,
                    max_retries=self._settings.llm_max_retries,
                )
            # Any constructor failure (missing key, bad option) means this provider
            # cannot serve; typed as unavailable so the router tries the next one.
            except Exception as exc:
                raise ProviderUnavailableError(f"openai init failed: {exc}") from exc
        return self._client

    async def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
    ) -> Completion:
        client = self._get_client()
        started = time.perf_counter()
        try:
            resp = await client.chat.completions.create(
                model=self._model,
                max_tokens=max_tokens,
                messages=[{"role": m.role.value, "content": m.content} for m in messages],
            )
        # Every vendor failure is translated to a typed Verity error and re-raised.
        except Exception as exc:
            _raise_translated(exc, sdk=_lazy_openai(), provider="openai")
        elapsed_ms = (time.perf_counter() - started) * 1000
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = getattr(resp, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        return Completion(
            text=text,
            provider=self.provider,
            model=self._model,
            usage=UsageStats(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                llm_calls=1,
                cost_usd=self.cost_usd(input_tokens=input_tokens, output_tokens=output_tokens),
                latency_ms=round(elapsed_ms, 2),
            ),
        )


# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------


def _split_system(messages: list[Message]) -> tuple[str, list[Message]]:
    """Separate the (single) system message — Anthropic wants it as a top-level arg."""
    system_parts: list[str] = []
    chat: list[Message] = []
    for m in messages:
        if m.role.value == "system":
            system_parts.append(m.content)
        else:
            chat.append(m)
    return ("\n\n".join(system_parts), chat)


def _unwrap(key: SecretStr | None) -> str | None:
    """Expose the raw key only at the SDK constructor; ``None`` lets the SDK fall
    back to its own environment lookup, as before."""
    return key.get_secret_value() if key is not None else None


def _raise_translated(exc: BaseException, *, sdk: Any, provider: str) -> NoReturn:
    """Raise the Verity error for an SDK exception. Never returns.

    Mapping by class, most specific first (both SDKs expose the same hierarchy):
    401/403 → AuthenticationError (no fallback: a bad key is a configuration bug);
    timeout, connection, 429 and 5xx → ProviderUnavailableError (fallback);
    anything else, including other 4xx → LLMError (a request bug, not retried).
    """
    if isinstance(exc, (sdk.AuthenticationError, sdk.PermissionDeniedError)):
        raise AuthenticationError(f"{provider}: {exc}") from exc
    if isinstance(exc, (sdk.APITimeoutError, sdk.APIConnectionError, sdk.RateLimitError)):
        raise ProviderUnavailableError(f"{provider}: {exc}") from exc
    if isinstance(exc, sdk.APIStatusError) and exc.status_code >= 500:
        raise ProviderUnavailableError(f"{provider}: {exc}") from exc
    raise LLMError(f"{provider}: {exc}") from exc


__all__ = [
    "AnthropicClient",
    "OpenAIClient",
    "Responder",
    "StubLLMClient",
]
