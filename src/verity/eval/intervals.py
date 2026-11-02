"""Wilson score interval for a published proportion.

The normal approximation p ± z·sqrt(p(1-p)/n) collapses to [0, 0] at 0/n and leaves
[0, 1] at small n; Wilson's interval does neither, which matters at the sizes this
eval publishes (0/31 hallucinations, 17/22 refusals). It assumes independent trials:
for per-claim ratios, claims of one answer are correlated, so the interval is too
narrow there (docs/evaluation-methodology.md says so next to the figures).
"""

from __future__ import annotations

import math

Z_95 = 1.959963984540054


def wilson(successes: int, n: int, z: float = Z_95) -> tuple[float, float] | None:
    """``(low, high)`` for ``successes`` out of ``n``; ``None`` when ``n`` is 0."""
    if isinstance(successes, float) and not successes.is_integer():
        raise ValueError(f"successes must be a count, got {successes}")
    if n < 0 or not 0 <= successes <= n:
        raise ValueError(f"need 0 <= successes <= n, got {successes}/{n}")
    if n == 0:
        return None
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    # Clamp the rounding residue at the bounds (0/n gives -1e-17, not 0).
    return max(0.0, centre - half), min(1.0, centre + half)
