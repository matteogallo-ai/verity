"""Threshold sweep and calibration metrics, on hand-computed examples."""

from __future__ import annotations

import pytest

from verity.eval.calibration import (
    ScoredExample,
    calibration_pairs,
    choose_threshold,
    expected_calibration_error,
    refusal_point,
    reliability_bins,
    sweep,
)
from verity.types import RefusalReason

_GATE = RefusalReason.NO_CITATION
_LOW = RefusalReason.BELOW_THRESHOLD

# Two unanswerable (one stopped by a deterministic check, one scored 0.3) and three
# answerable (scored 0.4, 0.7, 0.9).
_EXAMPLES = [
    ScoredExample(expected_answerable=False, score=0.0, reason=_GATE),
    ScoredExample(expected_answerable=False, score=0.3, reason=_LOW),
    ScoredExample(expected_answerable=True, score=0.4, reason=_LOW),
    ScoredExample(expected_answerable=True, score=0.7, reason=None),
    ScoredExample(expected_answerable=True, score=0.9, reason=None),
]


def test_a_deterministic_refusal_holds_at_every_threshold() -> None:
    assert _EXAMPLES[0].refused_at(0.0)
    assert not _EXAMPLES[4].refused_at(0.5)


@pytest.mark.parametrize(
    ("threshold", "tp", "fp", "fn", "precision", "recall"),
    [
        (0.0, 1, 0, 1, 1.0, 0.5),  # only the deterministic refusal
        (0.35, 2, 0, 0, 1.0, 1.0),  # 0.3 now refused, 0.4 answered
        (0.5, 2, 1, 0, 2 / 3, 1.0),  # 0.4 (answerable) now refused too
    ],
)
def test_refusal_point_counts_the_confusion_matrix(
    threshold: float, tp: int, fp: int, fn: int, precision: float, recall: float
) -> None:
    point = refusal_point(_EXAMPLES, threshold)
    assert (point.tp, point.fp, point.fn) == (tp, fp, fn)
    assert point.precision == pytest.approx(precision)
    assert point.recall == pytest.approx(recall)


def test_precision_is_none_when_nothing_is_refused() -> None:
    answered = [ScoredExample(expected_answerable=True, score=0.9, reason=None)]
    assert refusal_point(answered, 0.5).precision is None


def test_choose_threshold_maximises_recall_under_the_precision_floor() -> None:
    points = sweep(_EXAMPLES, [0.0, 0.35, 0.5])
    assert choose_threshold(points, min_precision=0.9).threshold == 0.35


def test_choose_threshold_refuses_to_guess_when_no_point_qualifies() -> None:
    with pytest.raises(ValueError, match="no threshold"):
        choose_threshold(sweep(_EXAMPLES, [0.5]), min_precision=0.9)


def test_calibration_uses_only_scorer_scores() -> None:
    assert calibration_pairs(_EXAMPLES) == [(0.3, False), (0.4, True), (0.7, True), (0.9, True)]


def test_ece_by_hand() -> None:
    # Bins: [0.3] -> |0.3-0|; [0.4] -> |0.4-1|; [0.7] -> |0.7-1|; [0.9] -> |0.9-1|.
    pairs = [(0.3, False), (0.4, True), (0.7, True), (0.9, True)]
    expected = (0.3 + 0.6 + 0.3 + 0.1) / 4
    assert expected_calibration_error(pairs) == pytest.approx(expected)


def test_a_perfectly_calibrated_bin_has_zero_error_and_one_point_zero_lands_in_the_last_bin() -> (
    None
):
    pairs = [(0.95, True), (1.0, True)]
    bins = reliability_bins(pairs)
    assert bins[-1].n == 2
    assert expected_calibration_error([(0.5, True), (0.5, False)]) == pytest.approx(0.0)


def test_ece_of_nothing_is_none() -> None:
    assert expected_calibration_error([]) is None
