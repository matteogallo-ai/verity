"""Micro and macro answer quality, checked by hand on four audit records."""

from __future__ import annotations

import pytest

from verity.eval.answer_quality import answer_quality
from verity.types import AnswerMetrics, ClaimVerdict, PerExampleAudit


def _audit(
    answerable: bool, claims: tuple[ClaimVerdict, ...] | None, *, refused: bool = False
) -> PerExampleAudit:
    """Per-answer metrics computed the way the LLM judge computes them."""
    if claims is None or not claims:
        faith = cite = 1.0 if claims is None else 0.0
        hall = 1.0 if claims == () and answerable else 0.0
    else:
        faith = sum(c.supported for c in claims) / len(claims)
        cite = sum(c.citation_ok for c in claims) / len(claims)
        hall = 1.0 if answerable and faith < 1.0 else 0.0
    return PerExampleAudit(
        example_id="q",
        question="?",
        expected_answerable=answerable,
        answer_text="",
        refused=refused,
        judge_claims=claims,
        answer_metrics=AnswerMetrics(
            faithfulness=faith,
            citation_accuracy=cite,
            hallucination_rate=hall,
            refusal_precision=0.0,
            refusal_recall=0.0,
        ),
    )


_OK = ClaimVerdict(claim="a", supported=True, citation_ok=True)
_UNSUPPORTED = ClaimVerdict(claim="b", supported=False, citation_ok=False)
_BAD_CITATION = ClaimVerdict(claim="c", supported=True, citation_ok=False)

_AUDITS = [
    _audit(True, (_OK, _OK, _OK, _UNSUPPORTED)),  # 3/4 supported, 3/4 cited
    _audit(True, (_BAD_CITATION,)),  # 1/1 supported, 0/1 cited
    _audit(True, ()),  # no usable claim: 0 in macro, nothing in micro
    _audit(False, None, refused=True),  # refusal: bypasses the judge
]


def test_macro_is_per_question_and_micro_is_per_claim() -> None:
    q = answer_quality(_AUDITS)
    assert (q.n_judged, q.n_judged_without_claims) == (3, 1)
    assert q.faithfulness_macro.value == pytest.approx((0.75 + 1.0 + 0.0) / 3)
    assert q.faithfulness_micro.value == pytest.approx(4 / 5)
    assert q.faithfulness_micro.denominator == 5
    assert q.citation_accuracy_macro.value == pytest.approx((0.75 + 0.0 + 0.0) / 3)
    assert q.citation_accuracy_micro.value == pytest.approx(3 / 5)
    # At risk: the three answered answerable questions; the first and the claimless one fail.
    assert q.hallucination_rate.value == pytest.approx(2 / 3)


def test_nothing_judged_gives_no_number() -> None:
    q = answer_quality([_audit(False, None, refused=True)])
    assert q.faithfulness_macro.value is None
    assert q.faithfulness_micro.value is None
    assert q.hallucination_rate.denominator == 0


def test_invented_answer_rate_counts_every_shipped_answer() -> None:
    """An unsupported claim is an invention whether or not the question had an answer:
    an answer shipped to an unanswerable question (q-074 on test) is in the denominator,
    and counts when one of its claims is unsupported."""
    audits = [
        *_AUDITS,
        _audit(False, (_OK, _UNSUPPORTED)),  # answered an unanswerable question, invented
        _audit(False, (_OK,)),  # answered an unanswerable question, every claim supported
    ]
    q = answer_quality(audits)
    # Judged: 5 answers. Invented: the 3/4 answer, the claimless one, the unanswerable one.
    assert (q.invented_answer_rate.numerator, q.invented_answer_rate.denominator) == (3, 5)
    # The answerable-only rate is unchanged by the two unanswerable answers.
    assert q.hallucination_rate.value == pytest.approx(2 / 3)
