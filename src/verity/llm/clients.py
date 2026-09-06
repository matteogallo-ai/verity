"""LLMClient implementations for Anthropic, OpenAI, and a deterministic stub.

The real clients wrap each vendor's async SDK behind the ``LLMClient`` protocol and
translate vendor-specific error shapes into the two Verity errors the routing client
branches on (``AuthenticationError`` short-circuits; ``ProviderUnavailableError``
triggers fallback). Any other exception propagates unchanged.

``StubLLMClient`` is what makes the eval harness and CI free: it takes a
``responder`` callable, deterministically maps a message list to a completion, and
never touches a network. Every S3+ test constructs the agent with a stub — no keys
required to prove the pipeline works.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

from verity.config import get_settings
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
        temperature: float = 0.0,
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
    """Anthropic Messages API. Model + key resolved from config/env; ``0.0`` temperature
    keeps eval-mode outputs stable."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        cost_per_input_1k: float = 0.003,
        cost_per_output_1k: float = 0.015,
    ) -> None:
        settings = get_settings()
        self._api_key = api_key
        self._model = model or settings.anthropic_model
        self._cost_in = cost_per_input_1k
        self._cost_out = cost_per_output_1k
        self._client: Any = None

    @property
    def provider(self) -> Provider:
        return Provider.ANTHROPIC

    @property
    def model(self) -> str:
        return self._model

    def _get_client(self) -> Any:
        if self._client is None:
            anthropic = _lazy_anthropic()
            try:
                self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
            except Exception as exc:
                raise ProviderUnavailableError(f"anthropic init failed: {exc}") from exc
        return self._client

    async def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Completion:
        client = self._get_client()
        started = time.perf_counter()
        system, chat = _split_system(messages)
        try:
            resp = await client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": m.role.value, "content": m.content} for m in chat],
            )
        except Exception as exc:  # translate vendor errors
            _raise_translated(exc, provider="anthropic")
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
                cost_usd=(input_tokens / 1000.0) * self._cost_in
                + (output_tokens / 1000.0) * self._cost_out,
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
        api_key: str | None = None,
        model: str | None = None,
        cost_per_input_1k: float = 0.005,
        cost_per_output_1k: float = 0.015,
    ) -> None:
        settings = get_settings()
        self._api_key = api_key
        self._model = model or settings.openai_model
        self._cost_in = cost_per_input_1k
        self._cost_out = cost_per_output_1k
        self._client: Any = None

    @property
    def provider(self) -> Provider:
        return Provider.OPENAI

    @property
    def model(self) -> str:
        return self._model

    def _get_client(self) -> Any:
        if self._client is None:
            openai = _lazy_openai()
            try:
                self._client = openai.AsyncOpenAI(api_key=self._api_key)
            except Exception as exc:
                raise ProviderUnavailableError(f"openai init failed: {exc}") from exc
        return self._client

    async def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Completion:
        client = self._get_client()
        started = time.perf_counter()
        try:
            resp = await client.chat.completions.create(
                model=self._model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": m.role.value, "content": m.content} for m in messages],
            )
        except Exception as exc:
            _raise_translated(exc, provider="openai")
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
                cost_usd=(input_tokens / 1000.0) * self._cost_in
                + (output_tokens / 1000.0) * self._cost_out,
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


def _raise_translated(exc: BaseException, *, provider: str) -> None:
    """Translate a vendor-specific error into a Verity LLMError. Never returns."""
    name = type(exc).__name__.lower()
    if "authentication" in name or "permission" in name or "unauthorized" in name:
        raise AuthenticationError(f"{provider}: {exc}") from exc
    if (
        "ratelimit" in name
        or "rate_limit" in name
        or "timeout" in name
        or "connection" in name
        or "unavailable" in name
        or "apistatus" in name
        or "internalserver" in name
    ):
        raise ProviderUnavailableError(f"{provider}: {exc}") from exc
    raise LLMError(f"{provider}: {exc}") from exc


# tests use this to prove routing calls happen concurrently or serially as configured
async def _sleep(seconds: float) -> None:  # pragma: no cover — trivial
    await asyncio.sleep(seconds)


__all__ = [
    "AnthropicClient",
    "OpenAIClient",
    "Responder",
    "StubLLMClient",
]
