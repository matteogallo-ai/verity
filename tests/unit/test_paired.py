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


def test_power_is_one_when_every_pair_is_discordant_one_way_and_n_allows_rejection() -> None:
    from verity.eval.paired import mcnemar_power

    # 6 pairs all won by one arm: p = 2 / 2**6 = 0.03125 < 0.05, with certainty.
    assert mcnemar_power(6, 1.0, 0.0, alpha=0.05) == 1.0
    # 5 such pairs: smallest p is 0.0625, never below 0.05.
    assert mcnemar_power(5, 1.0, 0.0, alpha=0.05) == 0.0


def test_power_under_no_difference_is_at_most_alpha() -> None:
    from verity.eval.paired import mcnemar_power

    assert mcnemar_power(60, 0.1, 0.1, alpha=0.05) <= 0.05


def test_power_matches_a_hand_sum_on_a_small_case() -> None:
    from verity.eval.paired import mcnemar_power

    # n = 6, one arm wins each pair with probability 0.5, else concordant: rejection
    # needs b = 6 (p = 0.03125); b = 5, c = 0 gives p = 0.0625. P(b = 6) = 0.5**6.
    assert mcnemar_power(6, 0.5, 0.0, alpha=0.05) == pytest.approx(0.5**6)
