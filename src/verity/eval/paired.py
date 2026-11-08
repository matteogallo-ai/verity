"""Paired comparison of two arms run on the same questions: the exact McNemar test.

Two arms answering the same questions are paired, so whether their Wilson intervals
overlap is the wrong question: it ignores the pairing and is far more conservative
than a 5 % test (two disjoint 95 % intervals is roughly p < 0.006). McNemar uses
only the discordant questions, those where one arm succeeds and the other fails;
the exact version is a two-sided binomial test of those counts at p = 0.5, which
holds at the small n of this repository where the chi-square approximation does not.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import comb


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for ``b`` and ``c`` discordant pairs."""
    if b < 0 or c < 0:
        raise ValueError("discordant counts cannot be negative")
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return float(min(1.0, 2 * tail))


def mcnemar_power(n: int, p_a_only: float, p_b_only: float, *, alpha: float) -> float:
    """Probability that the exact McNemar test rejects at ``alpha`` on ``n`` pairs, when
    each pair independently goes to arm A only with ``p_a_only``, to arm B only with
    ``p_b_only`` and is concordant otherwise. Exact sum over the trinomial, no
    normal approximation: at n below 100 the approximation overstates power."""
    if p_a_only < 0 or p_b_only < 0 or p_a_only + p_b_only > 1:
        raise ValueError("discordance probabilities must be in [0, 1] and sum to at most 1")
    p_same = 1.0 - p_a_only - p_b_only
    power = 0.0
    for b in range(n + 1):
        for c in range(n - b + 1):
            if mcnemar_exact(b, c) < alpha:
                power += (
                    comb(n, b) * comb(n - b, c) * p_a_only**b * p_b_only**c * p_same ** (n - b - c)
                )
    return power


def holm(pvalues: list[float]) -> list[float]:
    """Holm-adjusted p-values, in the input order: the family-wise error of the
    pre-registered primary tests stays at the level each adjusted p is compared to."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def discordant(a: Mapping[str, bool], b: Mapping[str, bool]) -> tuple[int, int, int]:
    """``(a only, b only, n paired)`` over the questions both mappings score."""
    shared = sorted(set(a) & set(b))
    a_only = sum(1 for q in shared if a[q] and not b[q])
    b_only = sum(1 for q in shared if b[q] and not a[q])
    return a_only, b_only, len(shared)


__all__ = ["discordant", "holm", "mcnemar_exact", "mcnemar_power"]
