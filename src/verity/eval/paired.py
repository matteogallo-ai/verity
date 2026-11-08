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


def discordant(a: Mapping[str, bool], b: Mapping[str, bool]) -> tuple[int, int, int]:
    """``(a only, b only, n paired)`` over the questions both mappings score."""
    shared = sorted(set(a) & set(b))
    a_only = sum(1 for q in shared if a[q] and not b[q])
    b_only = sum(1 for q in shared if b[q] and not a[q])
    return a_only, b_only, len(shared)


__all__ = ["discordant", "mcnemar_exact"]
