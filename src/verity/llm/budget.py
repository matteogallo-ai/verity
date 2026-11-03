"""A spending cap on a routing client, enforced before each call.

The cap is checked *before* a call, against spend so far plus an upper estimate of
that call, because a call's real cost is only known once it has been paid for.
Spend is the sum of the ``usage.cost_usd`` the clients report, i.e. the same
token counts and price table (``Settings.model_prices``) the eval harness sums.

Both cover the attempt that returns a completion. An attempt that fails (an SDK
retry, a routing fallback after a timeout) reports no usage, so if a provider bills
it anyway it is neither estimated nor counted.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from verity.llm.base import LLMError, RoutingClient
from verity.types import Completion, Message, UsageStats

# Chat templates add special tokens around each message (role markers, separators).
# OpenAI documents 3-4 per message plus 3 per reply; Anthropic does not publish its
# count, so 16 is an assumed margin, not a measured one.
MESSAGE_OVERHEAD_TOKENS = 16


@runtime_checkable
class _Priced(Protocol):
    """The price method :class:`AnthropicClient` and :class:`OpenAIClient` expose."""

    def cost_usd(self, *, input_tokens: int, output_tokens: int) -> float: ...


class SpendCapExceededError(LLMError):
    """The next LLM call could take spend past the cap, so it was not sent."""


def upper_cost_estimate(
    clients: Sequence[object], messages: list[Message], *, max_tokens: int
) -> float:
    """An upper estimate of the cost of one call, in USD.

    Input: a byte-level BPE tokenizer never emits a token shorter than one byte, so
    the UTF-8 length bounds the token count. That holds for OpenAI's (tiktoken);
    Anthropic's tokenizer is not public, so for it this is an assumption. Output:
    ``max_tokens`` is a hard limit the provider enforces. The dearest client counts because a
    routing fallback may serve the call. A client without a ``cost_usd`` price
    (the stubs) costs nothing.
    """
    input_tokens = sum(len(m.content.encode("utf-8")) + MESSAGE_OVERHEAD_TOKENS for m in messages)
    costs = [
        client.cost_usd(input_tokens=input_tokens, output_tokens=max_tokens)
        for client in clients
        if isinstance(client, _Priced)
    ]
    return float(max(costs, default=0.0))


class SpendCappedClient(RoutingClient):
    """Fronts a routing client; refuses any call that could take spend past ``cap_usd``.

    ``spent`` is the running total of every completion served through this wrapper.
    One instance caps one budget: build one per answer for a per-answer cap.
    """

    def __init__(self, inner: RoutingClient, *, cap_usd: float) -> None:
        if cap_usd <= 0:
            raise ValueError("cap_usd must be positive")
        self._inner = inner
        self._cap_usd = cap_usd
        self.spent = UsageStats()

    @property
    def clients(self) -> tuple[object, ...]:
        """The wrapped router's clients, so provenance and estimates see through the cap."""
        return tuple(getattr(self._inner, "clients", ()))

    async def complete(
        self, messages: list[Message], *, max_tokens: int = 1024, temperature: float = 0.0
    ) -> Completion:
        estimate = upper_cost_estimate(self.clients, messages, max_tokens=max_tokens)
        if self.spent.cost_usd + estimate > self._cap_usd:
            raise SpendCapExceededError(
                f"spent ${self.spent.cost_usd:.4f} of the ${self._cap_usd:.4f} cap; the next "
                f"call could cost up to ${estimate:.4f}, so it was not sent"
            )
        completion = await self._inner.complete(
            messages, max_tokens=max_tokens, temperature=temperature
        )
        used = completion.usage
        self.spent = UsageStats(
            input_tokens=self.spent.input_tokens + used.input_tokens,
            output_tokens=self.spent.output_tokens + used.output_tokens,
            llm_calls=self.spent.llm_calls + used.llm_calls,
            cost_usd=self.spent.cost_usd + used.cost_usd,
            retries=self.spent.retries + used.retries,
        )
        return completion


__all__ = [
    "MESSAGE_OVERHEAD_TOKENS",
    "SpendCapExceededError",
    "SpendCappedClient",
    "upper_cost_estimate",
]
