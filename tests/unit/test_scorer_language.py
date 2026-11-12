"""VERITY_LANGUAGE=fr: the scorer writes its rationale in French.

The rationale is shown under a refused answer; a French refusal over an English
reason reads as a broken page in front of a French-speaking member.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator

import pytest

from verity.agent.scorer import LLMConfidenceScorer
from verity.config import get_settings
from verity.types import Completion, Message, Provider, UsageStats


class _Client:
    def __init__(self) -> None:
        self.messages: list[Message] = []

    async def complete(self, messages: list[Message], *, max_tokens: int = 256) -> Completion:
        self.messages = messages
        text = json.dumps({"score": 0.9, "refused": False, "rationale": "ok"})
        return Completion(text=text, provider=Provider.ANTHROPIC, model="m", usage=UsageStats())


@pytest.fixture(autouse=True)
def _fresh_settings() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


async def _prompt(monkeypatch: pytest.MonkeyPatch, language: str | None) -> list[Message]:
    if language is None:
        monkeypatch.delenv("VERITY_LANGUAGE", raising=False)
    else:
        monkeypatch.setenv("VERITY_LANGUAGE", language)
    client = _Client()
    await LLMConfidenceScorer(client).score("q?", "a.", [])  # type: ignore[arg-type]
    return client.messages


@pytest.mark.asyncio
async def test_in_french_the_scorer_is_asked_for_a_french_rationale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system = (await _prompt(monkeypatch, "fr"))[0].content
    assert "Write the rationale in French" in system


# sha256 of the English scorer messages for ("q?", "a.", no evidence), taken before the
# language note existed: the published runs were scored with exactly this prompt.
_ENGLISH_PROMPT_SHA256 = "e0bd877c78a803e1c54a0ddb799f29c23975091a48b0a1085bde33e28799353b"


@pytest.mark.asyncio
@pytest.mark.parametrize("language", [None, "en"])
async def test_in_english_the_prompt_is_the_one_the_published_runs_used(
    monkeypatch: pytest.MonkeyPatch, language: str | None
) -> None:
    messages = await _prompt(monkeypatch, language)
    joined = "\x00".join(f"{m.role}:{m.content}" for m in messages)
    assert hashlib.sha256(joined.encode()).hexdigest() == _ENGLISH_PROMPT_SHA256
