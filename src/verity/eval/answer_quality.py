"""Answer-quality metrics from per-example audit records: the one definition.

The harness scorecard, the published headline and ``docs/evaluation-methodology.md``
all use this module, so a number cannot differ between the artefact and the README.

- *Judged* answers are those the judge saw (``judge_claims is not None``); refusals
  bypass the judge and are in no denominator here.
- *Unparseable*: the judge answered but its output could not be parsed
  (``judge_claims == ()``). The judge scores such an answer 0, so it counts as 0 in
  the macro figures and contributes no claim to the micro figures; its count is
  published so the two can be reconciled.
- **Macro** (per question): mean of the per-answer ratio, over judged answers.
- **Micro** (per claim): claims judged supported (or whose citation supports them)
  over all claims, pooled across parsed answers.
- Hallucination rate: share of answerable questions the agent answered whose
  answer has at least one unsupported claim (or unparseable judge output).

A ratio over an empty set is ``None``, never a made-up 0 or 1.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from verity.types import PerExampleAudit


@dataclass(frozen=True)
class Ratio:
    value: float | None
    numerator: float
    denominator: int


@dataclass(frozen=True)
class AnswerQuality:
    n_judged: int
    n_judge_unparseable: int
    faithfulness_macro: Ratio
    faithfulness_micro: Ratio
    citation_accuracy_macro: Ratio
    citation_accuracy_micro: Ratio
    hallucination_rate: Ratio


def _ratio(numerator: float, denominator: int) -> Ratio:
    return Ratio(numerator / denominator if denominator else None, numerator, denominator)


def answer_quality(audits: Sequence[PerExampleAudit]) -> AnswerQuality:
    judged = [a for a in audits if a.judge_claims is not None]
    claims = [c for a in judged for c in a.judge_claims or ()]
    at_risk = [a for a in judged if a.expected_answerable]
    return AnswerQuality(
        n_judged=len(judged),
        n_judge_unparseable=sum(1 for a in judged if not a.judge_claims),
        faithfulness_macro=_ratio(sum(a.answer_metrics.faithfulness for a in judged), len(judged)),
        faithfulness_micro=_ratio(sum(1 for c in claims if c.supported), len(claims)),
        citation_accuracy_macro=_ratio(
            sum(a.answer_metrics.citation_accuracy for a in judged), len(judged)
        ),
        citation_accuracy_micro=_ratio(sum(1 for c in claims if c.citation_ok), len(claims)),
        hallucination_rate=_ratio(
            sum(a.answer_metrics.hallucination_rate for a in at_risk), len(at_risk)
        ),
    )
