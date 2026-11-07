"""Refusal recall per tag, and unanswerable answered under two rules, by hand."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from verity.types import AnswerMetrics, ClaimVerdict, PerExampleAudit

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "refusal_by_tag.py"
_METRICS = AnswerMetrics(
    faithfulness=1.0,
    citation_accuracy=1.0,
    hallucination_rate=0.0,
    refusal_precision=0.0,
    refusal_recall=0.0,
)


def _module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    spec = importlib.util.spec_from_file_location("refusal_by_tag", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "refusal_by_tag", module)
    spec.loader.exec_module(module)
    return module


def _audit(qid: str, answerable: bool, refused: bool, *, supported: bool = True) -> PerExampleAudit:
    claims = None if refused else (ClaimVerdict(claim="c", supported=supported, citation_ok=True),)
    return PerExampleAudit(
        example_id=qid,
        question="?",
        expected_answerable=answerable,
        answer_text="" if refused else "a",
        refused=refused,
        judge_claims=claims,
        answer_metrics=_METRICS,
    )


_TAGS = {
    "u1": ("out-of-corpus",),
    "u2": ("adversarial", "out-of-corpus"),
    "u3": ("near-miss",),
    "u4": ("false-premise",),
    "u5": ("near-miss",),
    "a1": ("numeric",),
}
_AUDITS = [
    _audit("u1", False, True),
    _audit("u2", False, True),
    _audit("u3", False, False, supported=False),  # answered, invented
    _audit("u4", False, False),  # answered, every claim supported (a premise correction)
    _audit("u5", False, True),
    _audit("a1", True, True),
]


def test_easy_questions_are_split_out_of_the_recall(monkeypatch: pytest.MonkeyPatch) -> None:
    report = _module(monkeypatch).analyse(_AUDITS, _TAGS)["refusal_recall"]
    assert (report["all"]["k"], report["all"]["n"]) == (3, 5)
    assert (report["without_out_of_corpus"]["k"], report["without_out_of_corpus"]["n"]) == (1, 3)
    hard = report["without_out_of_corpus_and_adversarial"]
    assert (hard["k"], hard["n"], hard["answered"]) == (1, 3, ["u3", "u4"])
    assert (report["by_tag"]["near-miss"]["k"], report["by_tag"]["near-miss"]["n"]) == (1, 2)


def test_unanswerable_answered_under_both_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    ua = _module(monkeypatch).analyse(_AUDITS, _TAGS)["unanswerable_answered"]
    assert (ua["rule_of_record"]["k"], ua["rule_of_record"]["n"]) == (2, 5)
    alt = ua["fails_only_if_invented"]
    assert (alt["k"], alt["n"], alt["invented"], alt["supported"]) == (1, 5, ["u3"], ["u4"])


def test_a_run_question_missing_from_the_dataset_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(SystemExit, match="u1"):
        _module(monkeypatch).analyse(_AUDITS, {k: v for k, v in _TAGS.items() if k != "u1"})


def test_committed_outputs_match_the_run_of_record(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _module(monkeypatch).main(["--check"]) == 0
