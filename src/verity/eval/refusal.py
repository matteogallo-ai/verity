"""Refusal calibration — the headline metric of Verity's thesis, deterministic and LLM-free.

Framed as binary classification with "refuse" as the positive class:

- TP — refused an unanswerable question (correct)
- FP — refused an answerable question (uselessly timid)
- FN — answered an unanswerable question (hallucination)
- TN — answered an answerable question (correct)

Metrics (ADR 0003, docs/evaluation-methodology.md):

- ``refusal_recall`` = TP / (TP + FN) — of the questions that *should* have been
  refused, the fraction the system actually refused. Low recall = the system
  hallucinates on out-of-scope.
- ``refusal_precision`` = TP / (TP + FP) — of the questions the system refused,
  the fraction that genuinely should have been refused. Low precision = the
  system is uselessly timid on answerable questions.

Both are computed here without any LLM call, from the ``answerable`` label on the
example and ``Answer.confidence.refused`` on the observed output. This is why the
refusal chiffre in the Scorecard is real from day one, even in ``--judge stub`` mode.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class RefusalOutcome:
    """One (expected, observed) pair used to build the confusion matrix."""

    example_id: str
    expected_answerable: bool
    observed_refused: bool

    @property
    def tp(self) -> bool:  # refused an unanswerable → correct refusal
        return self.observed_refused and not self.expected_answerable

    @property
    def fp(self) -> bool:  # refused an answerable → uselessly timid
        return self.observed_refused and self.expected_answerable

    @property
    def fn(self) -> bool:  # answered an unanswerable → hallucination
        return (not self.observed_refused) and (not self.expected_answerable)

    @property
    def tn(self) -> bool:  # answered an answerable → correct answer
        return (not self.observed_refused) and self.expected_answerable


@dataclass(frozen=True)
class ConfusionMatrix:
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def total_should_refuse(self) -> int:
        return self.tp + self.fn

    @property
    def total_refusals(self) -> int:
        return self.tp + self.fp

    @property
    def refusal_recall(self) -> float:
        """Undefined when no example should be refused → convention 1.0 (a system
        that never faces an out-of-scope question hasn't been *tested* on refusal).
        The eval dataset always contains ≥1 out-of-scope by design, so in practice
        this branch is never taken; documented for API completeness."""
        return self.tp / self.total_should_refuse if self.total_should_refuse else 1.0

    @property
    def refusal_precision(self) -> float:
        """Undefined when the system never refused → convention 1.0. Combined with a
        recall of 0 this shape (never refuses) is what one would expect from a
        non-Verity system; the calibration story is in the pair, not in either
        number alone (see docs/evaluation-methodology.md)."""
        return self.tp / self.total_refusals if self.total_refusals else 1.0


def build_confusion(outcomes: Iterable[RefusalOutcome]) -> ConfusionMatrix:
    tp = fp = fn = tn = 0
    for o in outcomes:
        tp += 1 if o.tp else 0
        fp += 1 if o.fp else 0
        fn += 1 if o.fn else 0
        tn += 1 if o.tn else 0
    return ConfusionMatrix(tp=tp, fp=fp, fn=fn, tn=tn)


__all__ = ["ConfusionMatrix", "RefusalOutcome", "build_confusion"]
