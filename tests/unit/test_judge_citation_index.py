"""Judge prompt v0.2.0 names, per claim, the citation that supports it (finding M1 of
the audit): a citation can be correct for one claim of an answer and not for another.
The code checks the named index; a claim whose index is missing or out of range has
``citation_ok`` false whatever the model said."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from verity.eval.judge import LLMFaithfulnessJudge
from verity.eval.run_store import FileRunStore, load_run
from verity.types import (
    Answer,
    Chunk,
    Citation,
    ClaimVerdict,
    Completion,
    Confidence,
    EvalExample,
    Message,
    Provider,
    RetrievalHit,
    RetrieverKind,
)

_DOC = "00000000-0000-0000-0000-000000000001"


class _Fixed:
    def __init__(self, claims: list[dict[str, object]]) -> None:
        self.text = json.dumps({"claims": claims})

    async def complete(self, messages: list[Message], *, max_tokens: int = 1024) -> Completion:
        return Completion(text=self.text, provider=Provider.ANTHROPIC, model="fixed")


def _answer(n_citations: int) -> Answer:
    text = "Fees are due in 30 days. Late fees are 2%."
    chunk = Chunk(document_id=_DOC, text=text, ordinal=0, char_start=0, char_end=len(text))
    hit = RetrievalHit(chunk=chunk, score=1.0, kind=RetrieverKind.RERANKED, rank=0)
    citations = [
        Citation(chunk_id=chunk.id, document_id=_DOC, quote=text[:24], char_start=0, char_end=24)
    ] * n_citations
    return Answer(
        query="q",
        text=text,
        citations=citations,
        hits_used=[hit],
        confidence=Confidence(score=0.9, refused=False, rationale="r"),
    )


def _judge(claims: list[dict[str, object]]) -> LLMFaithfulnessJudge:
    # An answer has at least one citation here: an uncited one is a refusal (the Answer
    # type turns it into one) and never reaches the judge.
    judge = LLMFaithfulnessJudge(_Fixed(claims), model_label="fixed")
    example = EvalExample(id="q", question="q", answerable=True)
    asyncio.run(judge.judge(example, _answer(1)))
    return judge


def _claim(citation: object, *, citation_ok: bool = True, **extra: object) -> dict[str, object]:
    return {"claim": "c", "supported": True, "citation_ok": citation_ok, **extra} | (
        {} if citation is _MISSING else {"citation": citation}
    )


_MISSING = object()


def test_a_named_citation_in_range_keeps_the_verdict_and_is_recorded() -> None:
    (verdict,) = _judge([_claim(0)]).last_claims or ()
    assert verdict == ClaimVerdict(claim="c", supported=True, citation_ok=True, citation=0)


@pytest.mark.parametrize(
    "citation",
    [_MISSING, None, 1, -1, True, "0", 0.0],
    ids=["missing", "null", "out-of-range", "negative", "bool", "string", "float"],
)
def test_a_missing_or_invalid_index_makes_the_citation_not_ok(citation: object) -> None:
    (verdict,) = _judge([_claim(citation)]).last_claims or ()
    assert verdict.citation_ok is False
    assert verdict.supported is True  # support is judged on all the evidence


def test_the_model_saying_not_ok_is_kept_even_with_a_valid_index() -> None:
    (verdict,) = _judge([_claim(0, citation_ok=False)]).last_claims or ()
    assert (verdict.citation_ok, verdict.citation) == (False, 0)


def test_citation_accuracy_counts_the_checked_verdicts() -> None:
    judge = LLMFaithfulnessJudge(_Fixed([_claim(0), _claim(7)]), model_label="fixed")
    metrics = asyncio.run(
        judge.judge(EvalExample(id="q", question="q", answerable=True), _answer(1))
    )
    assert metrics.citation_accuracy == 0.5
    assert metrics.faithfulness == 1.0


def test_the_citation_index_round_trips_and_is_absent_when_unset(tmp_path: Path) -> None:
    from tests.unit.test_headline_generator import _fixture_result

    base = _fixture_result().as_eval_run()
    named = ClaimVerdict(claim="a", supported=True, citation_ok=True, citation=2)
    unnamed = ClaimVerdict(claim="b", supported=True, citation_ok=False)
    audit = base.per_audit[0].model_copy(update={"judge_claims": (named, unnamed)})
    run = base.model_copy(update={"per_audit": (audit, *base.per_audit[1:])})
    path = FileRunStore(tmp_path).save(run)
    stored = json.loads(path.read_text(encoding="utf-8"))["per_audit"][0]["judge_claims"]
    assert stored[0]["citation"] == 2
    assert "citation" not in stored[1]
    assert load_run(path).per_audit[0].judge_claims == (named, unnamed)
