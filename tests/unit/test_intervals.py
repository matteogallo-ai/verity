"""verity.eval.intervals: Wilson score interval for a proportion."""

from __future__ import annotations

import pytest

from verity.eval.intervals import wilson


def test_wilson_matches_the_closed_form() -> None:
    # 17/22 at 95%: centre (p + z²/2n) / (1 + z²/n), half-width z·sqrt(p(1-p)/n + z²/4n²) / (1 + z²/n).
    low, high = wilson(17, 22)
    assert low == pytest.approx(0.5656, abs=1e-4)
    assert high == pytest.approx(0.8988, abs=1e-4)


def test_wilson_is_not_degenerate_at_zero_or_all() -> None:
    # The normal approximation gives [0, 0] for 0/31; Wilson does not.
    low, high = wilson(0, 31)
    assert low == 0.0 and high == pytest.approx(0.1102, abs=1e-4)
    low, high = wilson(31, 31)
    assert low == pytest.approx(1 - 0.1102, abs=1e-4) and high == 1.0


def test_wilson_needs_a_sample() -> None:
    assert wilson(0, 0) is None


@pytest.mark.parametrize(("k", "n"), [(-1, 5), (6, 5), (2.5, 5)])
def test_wilson_rejects_what_is_not_a_count(k: float, n: int) -> None:
    with pytest.raises(ValueError):
        wilson(k, n)  # type: ignore[arg-type]
