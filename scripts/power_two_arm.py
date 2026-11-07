"""Power of the exact McNemar tests pre-registered in docs/preregistration/two-arm-comparison.md.

    uv run python scripts/power_two_arm.py

No model call. Prints, per primary test, the probability of rejecting at the Holm
first-step level (0.05 / 3, the strictest any of the three tests faces) for a grid of
discordance rates. The naive arm ships every draft, so on the two "answered" metrics a
question Verity answers and the naive arm does not needs a naive self-abstention; the
grids take that direction as zero.
"""

from __future__ import annotations

from verity.eval.paired import mcnemar_exact, mcnemar_power

ALPHA = 0.05 / 3
TESTS = [
    # (metric, n paired questions, [(naive only, Verity only), ...])
    ("Unanswerable answered", 33, [(0.3, 0.0), (0.5, 0.0), (0.7, 0.0), (0.85, 0.0)]),
    ("Answerable answered", 62, [(0.03, 0.0), (0.05, 0.0), (0.10, 0.0), (0.15, 0.0)]),
    ("Invented answer", 95, [(0.03, 0.0), (0.05, 0.0), (0.05, 0.01), (0.10, 0.01)]),
]


def smallest_one_way_rejection(alpha: float) -> int:
    """Discordant pairs, all won by one arm, needed for p < alpha."""
    k = 1
    while mcnemar_exact(k, 0) >= alpha:
        k += 1
    return k


def main() -> int:
    print(
        f"alpha = 0.05 / 3 = {ALPHA:.4f}; one-way discordant pairs needed: "
        f"{smallest_one_way_rejection(ALPHA)}\n"
    )
    print("| Test | n | P(naive only) | P(Verity only) | Expected discordant | Power |")
    print("|---|---|---|---|---|---|")
    for metric, n, grid in TESTS:
        for a, b in grid:
            power = mcnemar_power(n, a, b, alpha=ALPHA)
            print(f"| {metric} | {n} | {a:.2f} | {b:.2f} | {n * (a + b):.1f} | {power:.3f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
