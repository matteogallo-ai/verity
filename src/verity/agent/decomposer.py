"""Question decomposition via the ``decompose.prompt`` PromptLang source.

Contract: return a list of sub-questions in the order they should be answered. A
simple, single-fact question returns a one-element list — we **do not** force
decomposition (the prompt is explicit about it too), because forcing a split adds
noise for questions that don't need it and the eval harness measures decomposition
quality by whether it helped, not by whether it happened.

Robustness: if the LLM returns something other than a JSON array (rare with the
strict prompt but possible with weaker models), fall back to ``[question]`` rather
than crashing. Fail-open on parsing keeps the agent usable; the eval harness will
surface the quality hit.
"""

from __future__ import annotations

import json

from verity.agent.base import QuestionDecomposer
from verity.config import get_settings
from verity.llm.base import RoutingClient
from verity.llm.prompt import load_prompt
from verity.types import UsageStats


class LLMQuestionDecomposer(QuestionDecomposer):
    """Prompt-driven decomposer. Caps sub-questions at ``config.max_agent_steps``.

    ``last_usage`` exposes the token/cost accounting for the most recent
    :meth:`decompose` call so the agent can aggregate a truthful total onto the
    returned :class:`Answer` — protocols return only the primary output.
    """

    def __init__(
        self,
        client: RoutingClient,
        *,
        max_steps: int | None = None,
        prompt_name: str = "decompose",
    ) -> None:
        settings = get_settings()
        self._client = client
        self._max_steps = max_steps if max_steps is not None else settings.max_agent_steps
        self._prompt = load_prompt(prompt_name)
        self.last_usage: UsageStats | None = None

    async def decompose(self, question: str) -> list[str]:
        messages = self._prompt.render(question=question)
        completion = await self._client.complete(messages, max_tokens=256, temperature=0.0)
        self.last_usage = completion.usage
        sub_questions = _parse_sub_questions(completion.text) or [question]
        return sub_questions[: self._max_steps]


def _parse_sub_questions(text: str) -> list[str]:
    """Extract a JSON array of strings from ``text``. Returns ``[]`` on failure."""
    stripped = text.strip()
    if not stripped:
        return []
    # A generous parser: try full-text JSON first, then the first bracketed slice.
    for candidate in (stripped, _first_bracketed(stripped)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
            return [item.strip() for item in parsed if item.strip()]
    return []


def _first_bracketed(text: str) -> str:
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return ""
    return text[start : end + 1]


__all__ = ["LLMQuestionDecomposer"]
