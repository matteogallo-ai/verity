"""Refusal confusion matrix — hand-computed cases."""

from __future__ import annotations

from verity.eval.refusal import RefusalOutcome, build_confusion


def _outcome(_id: str, *, expected_answerable: bool, observed_refused: bool) -> RefusalOutcome:
    return RefusalOutcome(
        example_id=_id,
        expected_answerable=expected_answerable,
        observed_refused=observed_refused,
    )


def test_perfect_classifier() -> None:
    outcomes = [
        _outcome("q1", expected_answerable=True, observed_refused=False),  # TN
        _outcome("q2", expected_answerable=False, observed_refused=True),  # TP
    ]
    c = build_confusion(outcomes)
    assert c.tp == 1 and c.tn == 1 and c.fp == 0 and c.fn == 0
    assert c.refusal_recall == 1.0
    assert c.refusal_precision == 1.0


def test_hallucination_on_out_of_scope() -> None:
    outcomes = [
        _outcome("q1", expected_answerable=False, observed_refused=False),  # FN (hallucinated)
        _outcome("q2", expected_answerable=False, observed_refused=True),  # TP
    ]
    c = build_confusion(outcomes)
    assert c.tp == 1 and c.fn == 1
    assert c.refusal_recall == 0.5


def test_uselessly_timid_lowers_precision() -> None:
    outcomes = [
        _outcome("q1", expected_answerable=True, observed_refused=True),  # FP (timid)
        _outcome("q2", expected_answerable=False, observed_refused=True),  # TP
    ]
    c = build_confusion(outcomes)
    assert c.tp == 1 and c.fp == 1
    assert c.refusal_precision == 0.5
    assert c.refusal_recall == 1.0


def test_never_refuses_yields_zero_recall() -> None:
    outcomes = [
        _outcome("q1", expected_answerable=True, observed_refused=False),
        _outcome("q2", expected_answerable=False, observed_refused=False),
    ]
    c = build_confusion(outcomes)
    assert c.tp == 0 and c.fn == 1
    assert c.refusal_recall == 0.0
    # No refusals emitted → refusal_precision undefined → convention 1.0.
    assert c.refusal_precision == 1.0


def test_empty_outcomes_conventions() -> None:
    c = build_confusion([])
    assert c.refusal_recall == 1.0  # nothing to fail
    assert c.refusal_precision == 1.0
