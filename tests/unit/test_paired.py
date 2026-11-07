"""Exact McNemar test on paired binary outcomes, checked against hand values."""

from __future__ import annotations

import pytest

from verity.eval.paired import discordant, mcnemar_exact


@pytest.mark.parametrize(
    ("b", "c", "p"),
    [
        (0, 5, 0.0625),  # Verity 16/21 vs naive 21/21 answerable answered: 2 * 0.5**5
        (5, 0, 0.0625),
        (2, 8, 2 * (1 + 10 + 45) / 1024),
        (0, 0, 1.0),  # no discordant pair: no evidence either way
        (3, 3, 1.0),
    ],
)
def test_mcnemar_exact_two_sided(b: int, c: int, p: float) -> None:
    assert mcnemar_exact(b, c) == pytest.approx(p)


def test_mcnemar_rejects_negative_counts() -> None:
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 2)


def test_discordant_pairs_only_count_questions_both_arms_scored() -> None:
    a = {"q1": True, "q2": True, "q3": False, "q4": False}
    b = {"q1": True, "q2": False, "q3": True, "q5": True}
    # q2: a only; q3: b only; q1 concordant; q4, q5 in one arm only.
    assert discordant(a, b) == (1, 1, 3)
