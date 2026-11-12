"""VERITY_SYNTHESIS_MAX_TOKENS: the cited draft's output cap, for the API and the CLI.

A draft cut off at the cap ends without its citations and is refused as uncited; on
the hotel-restaurant CCT a complete answer on overtime pay took 862 output tokens,
over the default cap of 512.
"""

from __future__ import annotations

import json

import pytest

from verity.agent.synthesizer import DRAFT_MAX_TOKENS, CitedSynthesizer
from verity.config import get_settings
from verity.types import Completion, Message, Provider, UsageStats


class _Client:
    def __init__(self) -> None:
        self.max_tokens: int | None = None

    async def complete(self, messages: list[Message], *, max_tokens: int = 1024) -> Completion:
        self.max_tokens = max_tokens
        text = json.dumps({"answer": "x", "citations": []})
        return Completion(text=text, provider=Provider.ANTHROPIC, model="m", usage=UsageStats())


@pytest.fixture(autouse=True)
def _fresh_settings() -> object:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_the_setting_sets_the_cap_sent_to_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VERITY_SYNTHESIS_MAX_TOKENS", "1024")
    client = _Client()
    synthesizer = CitedSynthesizer(client)  # type: ignore[arg-type]
    await synthesizer.synthesize("q?", [])
    assert synthesizer.max_tokens == 1024
    assert client.max_tokens == 1024


def test_unset_the_cap_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VERITY_SYNTHESIS_MAX_TOKENS", raising=False)
    assert CitedSynthesizer(_Client()).max_tokens == DRAFT_MAX_TOKENS  # type: ignore[arg-type]


def test_an_explicit_cap_wins_over_the_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bench passes its own --draft-max-tokens."""
    monkeypatch.setenv("VERITY_SYNTHESIS_MAX_TOKENS", "1024")
    assert CitedSynthesizer(_Client(), max_tokens=300).max_tokens == 300  # type: ignore[arg-type]


@pytest.mark.parametrize("value", ["0", "-5"])
def test_a_cap_that_is_not_positive_is_rejected(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    from pydantic import ValidationError

    monkeypatch.setenv("VERITY_SYNTHESIS_MAX_TOKENS", value)
    with pytest.raises(ValidationError):
        get_settings()
