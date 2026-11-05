"""StubFaithfulnessJudge — deterministic verdicts on constructed cases."""

from __future__ import annotations

from uuid import UUID

import pytest

from verity.eval.judge import STUB_JUDGE_MODEL, StubFaithfulnessJudge
from verity.types import (
    Answer,
    Chunk,
    Citation,
    Confidence,
    DocumentId,
    EvalExample,
    RetrievalHit,
    RetrieverKind,
    UsageStats,
)

DOC_ID = DocumentId("00000000-0000-0000-0000-000000000000")


def _chunk(text: str, *, cs: int = 0, ordinal: int = 0) -> Chunk:
    return Chunk(
        id=UUID(int=ordinal + 1),
        document_id=DOC_ID,
        text=text,
        ordinal=ordinal,
        char_start=cs,
        char_end=cs + len(text),
    )


def _hit(text: str, *, ordinal: int = 0) -> RetrievalHit:
    return RetrievalHit(
        chunk=_chunk(text, cs=0, ordinal=ordinal),
        score=1.0,
        kind=RetrieverKind.RERANKED,
        rank=ordinal,
    )


def _answer(
    *,
    text: str,
    hits: list[RetrievalHit],
    citations: tuple[Citation, ...] = (),
    refused: bool = False,
) -> Answer:
    return Answer(
        query="q?",
        text=text,
        citations=citations,
        confidence=Confidence(
            score=0.1 if refused else 0.9,
            refused=refused,
            rationale="stub",
        ),
        hits_used=tuple(hits),
        usage=UsageStats(),
    )


def _example(answerable: bool) -> EvalExample:
    return EvalExample(
        id="q-x",
        question="q?",
        answerable=answerable,
        reference_answer=("ref" if answerable else None),
    )


@pytest.mark.asyncio
async def test_stub_judge_declares_itself() -> None:
    judge = StubFaithfulnessJudge()
    assert judge.model == STUB_JUDGE_MODEL


@pytest.mark.asyncio
async def test_stub_judge_refusal_scores_neutral() -> None:
    judge = StubFaithfulnessJudge()
    metrics = await judge.judge(
        _example(answerable=False),
        _answer(text="I don't know based on the provided documents.", hits=[], refused=True),
    )
    # Refusal → nothing to fail: faithfulness 1.0, hallucination 0.0.
    assert metrics.faithfulness == 1.0
    assert metrics.citation_accuracy == 1.0
    assert metrics.hallucination_rate == 0.0


@pytest.mark.asyncio
async def test_stub_judge_supported_answer_with_citation() -> None:
    judge = StubFaithfulnessJudge()
    chunk_text = "Either party may terminate on thirty (30) days notice."
    hit = _hit(chunk_text, ordinal=0)
    citation = Citation(
        chunk_id=hit.chunk.id,
        document_id=hit.chunk.document_id,
        quote="thirty (30) days",
        char_start=0,
        char_end=len("thirty (30) days"),
    )
    metrics = await judge.judge(
        _example(answerable=True),
        _answer(
            text="Termination requires thirty (30) days notice.",
            hits=[hit],
            citations=(citation,),
        ),
    )
    assert metrics.faithfulness == 1.0
    assert metrics.citation_accuracy == 1.0
    assert metrics.hallucination_rate == 0.0


@pytest.mark.asyncio
async def test_stub_judge_flags_hallucination_on_unsupported_claim() -> None:
    judge = StubFaithfulnessJudge()
    hit = _hit("Contract mentions widgets.", ordinal=0)  # short, unrelated context
    # A cited quote that does not support the claims: an uncited answer cannot
    # exist any more (Answer converts it into a no_citation refusal).
    citation = Citation(
        chunk_id=hit.chunk.id,
        document_id=hit.chunk.document_id,
        quote="widgets",
        char_start=hit.chunk.char_start + hit.chunk.text.find("widgets"),
        char_end=hit.chunk.char_start + hit.chunk.text.find("widgets") + len("widgets"),
    )
    metrics = await judge.judge(
        _example(answerable=True),
        _answer(
            text="The CEO's name is Alice Smith and revenue was $999 million.",
            hits=[hit],
            citations=(citation,),
        ),
    )
    assert metrics.hallucination_rate == 1.0
    assert metrics.faithfulness < 1.0


@pytest.mark.asyncio
async def test_stub_judge_records_zero_usage() -> None:
    judge = StubFaithfulnessJudge()
    await judge.judge(_example(answerable=True), _answer(text="short.", hits=[]))
    assert judge.last_usage is not None
    assert judge.last_usage.llm_calls == 0
    assert judge.last_usage.cost_usd == 0.0


@pytest.mark.asyncio
async def test_a_shipped_answer_that_says_it_does_not_know_is_judged() -> None:
    """Refusal is the agent's recorded decision, not a phrase: a shipped draft saying
    "I do not know" stays in the judged answers (naive baseline, pre-registered)."""
    judge = StubFaithfulnessJudge()
    shipped = Answer(
        query="q?",
        text="I do not know when the policy took effect.",
        confidence=Confidence(score=1.0, refused=False, rationale="checks disabled"),
        checks_disabled=True,
    )
    metrics = await judge.judge(_example(answerable=True), shipped)
    # The refusal shortcut would score it 1.0; judged with no citation it is 0.0.
    assert metrics.citation_accuracy == 0.0
