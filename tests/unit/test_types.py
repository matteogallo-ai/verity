"""Domain model invariants. These lock the contracts every stage depends on."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from verity.types import (
    Answer,
    AnswerMetrics,
    Chunk,
    Confidence,
    Document,
    EvalExample,
    LatencyMetrics,
    RetrievalMetrics,
    Scorecard,
    UsageStats,
)


def test_document_is_frozen(document: Document) -> None:
    with pytest.raises(ValidationError):
        document.text = "mutated"  # type: ignore[misc]


def test_chunk_offsets_are_preserved(chunk: Chunk) -> None:
    # Offsets are load-bearing for citation highlighting — assert they survive the model.
    assert chunk.char_end > chunk.char_start
    assert chunk.ordinal == 0


def test_confidence_score_is_bounded() -> None:
    with pytest.raises(ValidationError):
        Confidence(score=1.5, refused=False, rationale="x")
    with pytest.raises(ValidationError):
        Confidence(score=-0.1, refused=True, rationale="x")


def test_answer_defaults_are_populated(confident: Confidence) -> None:
    ans = Answer(query="q", text="a", confidence=confident)
    assert isinstance(ans.usage, UsageStats)
    assert ans.trace_id is not None
    assert ans.citations == ()


def test_refused_answer_carries_rationale(refused: Confidence) -> None:
    ans = Answer(query="q", text="I don't know.", confidence=refused)
    assert ans.confidence.refused is True
    assert ans.confidence.rationale


def test_eval_example_unanswerable_has_no_reference() -> None:
    ex = EvalExample(id="q-004", question="CEO home address?", answerable=False)
    assert ex.answerable is False
    assert ex.reference_answer is None


def test_scorecard_round_trips_json() -> None:
    card = Scorecard(
        git_sha="abc1234",
        dataset="default",
        n_examples=6,
        retrieval=RetrievalMetrics(precision_at_k=0.8, recall_at_k=0.7, ndcg_at_k=0.75, k=8),
        answer=AnswerMetrics(
            faithfulness=0.94,
            citation_accuracy=0.9,
            hallucination_rate=0.05,
            refusal_precision=0.88,
            refusal_recall=0.92,
        ),
        latency=LatencyMetrics(p50_ms=420.0, p95_ms=900.0, p99_ms=1300.0),
        cost_per_query_usd=0.0031,
    )
    restored = Scorecard.model_validate_json(card.model_dump_json())
    assert restored == card
    assert restored.git_sha == "abc1234"
