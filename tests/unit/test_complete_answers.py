"""VERITY_COMPLETE_ANSWERS: every part answered, the uncovered part named, figures exact.

The draft says what the documents do not cover in a separate ``not_covered`` field.
The agent appends it and the referral text to an answer that passed its checks: the
appended sentence is not document text, so it stays out of the citation-coverage
check that a sentence with no citation would otherwise drag down.
"""

from __future__ import annotations

import json
from uuid import UUID

import pytest

from verity.agent.agent import create_default_agent
from verity.agent.evidence import DATAMARK
from verity.agent.scripted_stub import build_stub_router
from verity.agent.synthesizer import CitedSynthesizer, SynthesisResult
from verity.types import (
    Chunk,
    Completion,
    DocumentId,
    Message,
    Provider,
    RetrievalHit,
    RetrieverKind,
    UsageStats,
)

_TEXT = "Art. 18 Le collaborateur a droit à 6 jours fériés payés par an."
_HIT = RetrievalHit(
    chunk=Chunk(
        id=UUID(int=1),
        document_id=DocumentId("00000000-0000-0000-0000-000000000000"),
        text=_TEXT,
        ordinal=0,
        char_start=0,
        char_end=len(_TEXT),
    ),
    score=1.0,
    kind=RetrieverKind.RERANKED,
    rank=0,
)
_REFERRAL = "Contactez le secrétariat de l'association."


class _Client:
    def __init__(self, payload: dict[str, object]) -> None:
        self.text = json.dumps(payload)
        self.messages: list[Message] = []

    async def complete(self, messages: list[Message], *, max_tokens: int = 1024) -> Completion:
        self.messages = messages
        return Completion(
            text=self.text, provider=Provider.ANTHROPIC, model="m", usage=UsageStats()
        )


def _payload(**extra: object) -> dict[str, object]:
    return {
        "answer": "6 jours fériés payés par an.",
        "citations": [{"quote": "6 jours fériés payés par an", "chunk_index": 0}],
        **extra,
    }


async def _synthesize(payload: dict[str, object], *, on: bool) -> tuple[SynthesisResult, str]:
    client = _Client(payload)
    result = await CitedSynthesizer(client, complete_answers=on).synthesize("q?", [_HIT])  # type: ignore[arg-type]
    return result, client.messages[0].content


@pytest.mark.asyncio
async def test_the_note_asks_for_every_part_the_missing_part_and_exact_figures() -> None:
    _, off = await _synthesize(_payload(), on=False)
    _, on = await _synthesize(_payload(), on=True)
    assert "not_covered" not in off
    assert on.startswith(off)
    assert '"not_covered"' in on
    assert "every part" in on
    assert "exactly as the cited text writes" in on


@pytest.mark.asyncio
async def test_the_uncovered_part_is_read_from_its_own_field() -> None:
    missing = f"Le texte ne{DATAMARK}dit rien du congé maternité."
    result, _ = await _synthesize(_payload(not_covered=missing), on=True)
    assert result.text == "6 jours fériés payés par an."
    assert result.not_covered == "Le texte ne dit rien du congé maternité."


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [None, "", "  ", 3, ["x"]])
async def test_an_empty_or_malformed_field_means_nothing_is_missing(value: object) -> None:
    result, _ = await _synthesize(_payload(not_covered=value), on=True)
    assert result.not_covered is None


@pytest.mark.asyncio
async def test_without_the_setting_the_field_is_ignored() -> None:
    result, _ = await _synthesize(_payload(not_covered="x"), on=False)
    assert result.not_covered is None


class _Synthesizer:
    max_tokens = 512

    def __init__(self, result: SynthesisResult) -> None:
        self._result = result

    async def synthesize(self, question: str, hits: list[RetrievalHit]) -> SynthesisResult:
        return self._result


class _Retriever:
    async def retrieve(self, query: str, *, k: int) -> list[RetrievalHit]:
        return [_HIT]


async def _answer(result: SynthesisResult, monkeypatch: pytest.MonkeyPatch, *, on: bool) -> str:
    from verity.config import get_settings

    monkeypatch.setenv("VERITY_COMPLETE_ANSWERS", "true" if on else "false")
    get_settings.cache_clear()
    agent = create_default_agent(
        retriever=_Retriever(),  # type: ignore[arg-type]
        client=build_stub_router(),
        synthesizer=_Synthesizer(result),  # type: ignore[arg-type]
        refusal_text=_REFERRAL,
        checks=False,
    )
    return (await agent.answer("q?")).text


def _result(not_covered: str | None, *, cited: bool = True) -> SynthesisResult:
    from verity.agent.synthesizer import _validate_citations

    citations = _validate_citations([("6 jours fériés payés par an", 0)], [_HIT]) if cited else ()
    return SynthesisResult(
        text="6 jours fériés payés par an.",
        citations=citations,
        usage=UsageStats(),
        provider=None,
        model=None,
        not_covered=not_covered,
    )


@pytest.mark.asyncio
async def test_an_answer_names_what_is_missing_then_refers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = await _answer(_result("Rien sur le congé maternité."), monkeypatch, on=True)
    assert text == f"6 jours fériés payés par an.\n\nRien sur le congé maternité. {_REFERRAL}"


@pytest.mark.asyncio
async def test_a_complete_answer_gets_no_referral(monkeypatch: pytest.MonkeyPatch) -> None:
    assert await _answer(_result(None), monkeypatch, on=True) == "6 jours fériés payés par an."


@pytest.mark.asyncio
async def test_without_the_setting_the_answer_is_the_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = await _answer(_result("Rien sur le congé maternité."), monkeypatch, on=False)
    assert text == "6 jours fériés payés par an."
