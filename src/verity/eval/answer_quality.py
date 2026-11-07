"""Answer-quality metrics from per-example audit records: the one definition.

The harness scorecard, the published headline and ``docs/evaluation-methodology.md``
all use this module, so a number cannot differ between the artefact and the README.

- *Judged* answers are those the judge saw (``judge_claims is not None``); refusals
  bypass the judge and are in no denominator here.
- *Without claims*: the judge saw the answer but returned no usable claim
  (``judge_claims == ()``): its output could not be parsed, or held no claim (the
  parser treats an empty claim list as malformed). The judge scores such an answer 0,
  so it counts as 0 in the macro figures and contributes no claim to the micro
  figures; its count is published so the two can be reconciled.
- **Macro** (per question): mean of the per-answer ratio, over judged answers.
- **Micro** (per claim): claims judged supported (or whose citation supports them)
  over all claims, pooled across judged answers.
- Hallucination rate: share of answerable questions the agent answered whose
  answer has at least one unsupported claim (or no usable claim from the judge).
- Invented-answer rate: the same test over *every* judged answer, answerable or not.
  An answer shipped to a question the corpus cannot answer is a user-visible answer
  too; when one of its claims is unsupported it is an invention, and the
  answerable-only rate cannot see it (q-074 on the test split).

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
    n_judged_without_claims: int
    faithfulness_macro: Ratio
    faithfulness_micro: Ratio
    citation_accuracy_macro: Ratio
    citation_accuracy_micro: Ratio
    hallucination_rate: Ratio
    invented_answer_rate: Ratio


def _invented(audit: PerExampleAudit) -> bool:
    """A judged answer with an unsupported claim, or with no usable claim: the judge
    could not confirm it, which the hallucination rate also counts as a failure."""
    claims = audit.judge_claims or ()
    return not claims or any(not c.supported for c in claims)


def _ratio(numerator: float, denominator: int) -> Ratio:
    return Ratio(numerator / denominator if denominator else None, numerator, denominator)


def answer_quality(audits: Sequence[PerExampleAudit]) -> AnswerQuality:
    judged = [a for a in audits if a.judge_claims is not None]
    claims = [c for a in judged for c in a.judge_claims or ()]
    at_risk = [a for a in judged if a.expected_answerable]
    return AnswerQuality(
        n_judged=len(judged),
        n_judged_without_claims=sum(1 for a in judged if not a.judge_claims),
        faithfulness_macro=_ratio(sum(a.answer_metrics.faithfulness for a in judged), len(judged)),
        faithfulness_micro=_ratio(sum(1 for c in claims if c.supported), len(claims)),
        citation_accuracy_macro=_ratio(
            sum(a.answer_metrics.citation_accuracy for a in judged), len(judged)
        ),
        citation_accuracy_micro=_ratio(sum(1 for c in claims if c.citation_ok), len(claims)),
        hallucination_rate=_ratio(
            sum(a.answer_metrics.hallucination_rate for a in at_risk), len(at_risk)
        ),
        invented_answer_rate=_ratio(sum(1 for a in judged if _invented(a)), len(judged)),
    )
