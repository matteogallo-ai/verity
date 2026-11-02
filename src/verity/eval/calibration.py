"""Refusal threshold sweep and calibration of the confidence score, from a run artefact.

Everything here is recomputed from :class:`~verity.types.PerExampleAudit` records, so a
threshold can be chosen on the dev run without calling a model again. That is valid
because the scorer prompt does not contain the threshold: the scorer's score is the
same whatever threshold the agent was run with.

Definitions (also in ``docs/evaluation-methodology.md``):

- A refusal is *threshold-independent* when a deterministic check or the scorer model
  itself refused (every reason except ``below_threshold``). At threshold ``t`` an
  example is refused iff it was refused threshold-independently or its score is
  below ``t``.
- Refusal is the positive class and an unanswerable question its target: refusal
  precision = TP / (TP + FP), refusal recall = TP / (TP + FN). ``None`` when the
  denominator is 0, never a made-up 0 or 1.
- Calibration reads the score as the probability that the question is answerable,
  on the examples the scorer actually scored (a deterministic refusal carries no
  score). ECE = sum over equal-width bins of |B|/n * |mean score(B) - answerable
  fraction(B)|.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from verity.eval.refusal import RefusalOutcome, build_confusion
from verity.types import PerExampleAudit, RefusalReason

_SCORER_REASONS = {None, RefusalReason.BELOW_THRESHOLD, RefusalReason.MODEL_REFUSED}


@dataclass(frozen=True)
class ScoredExample:
    expected_answerable: bool
    score: float
    reason: RefusalReason | None

    @property
    def scored_by_the_scorer(self) -> bool:
        return self.reason in _SCORER_REASONS

    def refused_at(self, threshold: float) -> bool:
        if self.reason not in (None, RefusalReason.BELOW_THRESHOLD):
            return True
        return self.score < threshold


@dataclass(frozen=True)
class RefusalPoint:
    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float | None
    recall: float | None


@dataclass(frozen=True)
class ReliabilityBin:
    lower: float
    upper: float
    n: int
    mean_score: float | None
    answerable_fraction: float | None


def scored_examples(audits: Sequence[PerExampleAudit]) -> list[ScoredExample]:
    """The audits as :class:`ScoredExample`; a run recorded without scores is refused."""
    examples: list[ScoredExample] = []
    for audit in audits:
        if audit.confidence_score is None:
            raise ValueError(
                f"{audit.example_id}: the run did not record confidence scores; "
                "the sweep needs a run made after they were stored"
            )
        examples.append(
            ScoredExample(audit.expected_answerable, audit.confidence_score, audit.refusal_reason)
        )
    return examples


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def refusal_point(examples: Sequence[ScoredExample], threshold: float) -> RefusalPoint:
    c = build_confusion(
        RefusalOutcome(str(i), ex.expected_answerable, ex.refused_at(threshold))
        for i, ex in enumerate(examples)
    )
    return RefusalPoint(
        threshold, c.tp, c.fp, c.fn, c.tn, _ratio(c.tp, c.tp + c.fp), _ratio(c.tp, c.tp + c.fn)
    )


def sweep(examples: Sequence[ScoredExample], thresholds: Sequence[float]) -> list[RefusalPoint]:
    return [refusal_point(examples, t) for t in thresholds]


def choose_threshold(points: Sequence[RefusalPoint], *, min_precision: float) -> RefusalPoint:
    """The point with the highest refusal recall among those whose refusal precision is
    at least ``min_precision``; ties go to the lowest threshold (fewest refusals of
    answerable questions). Without such a point, :class:`ValueError`: the choice is
    then a judgement call the caller must make and document."""
    eligible = [p for p in points if p.precision is not None and p.precision >= min_precision]
    if not eligible:
        raise ValueError(f"no threshold reaches refusal precision {min_precision}")
    return max(eligible, key=lambda p: (p.recall or 0.0, -p.threshold))


def best_f1_threshold(points: Sequence[RefusalPoint]) -> RefusalPoint:
    """The point with the highest refusal F1 (harmonic mean of refusal precision and
    recall); ties go to the lowest threshold. The fallback rule when no threshold
    reaches the precision floor of :func:`choose_threshold`."""

    def f1(p: RefusalPoint) -> float:
        if not p.precision or not p.recall:
            return 0.0
        return 2 * p.precision * p.recall / (p.precision + p.recall)

    return max(points, key=lambda p: (f1(p), -p.threshold))


def reliability_bins(
    pairs: Sequence[tuple[float, bool]], *, n_bins: int = 10
) -> list[ReliabilityBin]:
    """``pairs`` are ``(score, expected_answerable)``; the last bin includes 1.0."""
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for score, answerable in pairs:
        bins[min(int(score * n_bins), n_bins - 1)].append((score, answerable))
    out: list[ReliabilityBin] = []
    for i, members in enumerate(bins):
        n = len(members)
        out.append(
            ReliabilityBin(
                lower=i / n_bins,
                upper=(i + 1) / n_bins,
                n=n,
                mean_score=sum(s for s, _ in members) / n if n else None,
                answerable_fraction=sum(1 for _, a in members if a) / n if n else None,
            )
        )
    return out


def expected_calibration_error(
    pairs: Sequence[tuple[float, bool]], *, n_bins: int = 10
) -> float | None:
    if not pairs:
        return None
    total = len(pairs)
    return sum(
        b.n / total * abs(b.mean_score - b.answerable_fraction)
        for b in reliability_bins(pairs, n_bins=n_bins)
        if b.n and b.mean_score is not None and b.answerable_fraction is not None
    )


def calibration_pairs(examples: Sequence[ScoredExample]) -> list[tuple[float, bool]]:
    """``(score, expected_answerable)`` for the examples the scorer actually scored."""
    return [(ex.score, ex.expected_answerable) for ex in examples if ex.scored_by_the_scorer]
