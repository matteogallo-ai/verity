"""The registered per-question outcomes and decision sentences of the two-arm comparison."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from verity.types import AnswerMetrics, ClaimVerdict, PerExampleAudit

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "compare_two_arm.py"


def _module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    spec = importlib.util.spec_from_file_location("compare_two_arm", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "compare_two_arm", module)
    spec.loader.exec_module(module)
    return module


def _claim(supported: bool) -> ClaimVerdict:
    return ClaimVerdict(claim="c", supported=supported, citation_ok=supported)


def _audit(
    text: str, *, refused: bool = False, claims: tuple[ClaimVerdict, ...] | None = None
) -> PerExampleAudit:
    return PerExampleAudit(
        example_id="q",
        question="?",
        expected_answerable=False,
        answer_text=text,
        refused=refused,
        judge_claims=claims,
        # Not read by the outcomes; any valid value.
        answer_metrics=AnswerMetrics(
            faithfulness=1.0,
            citation_accuracy=1.0,
            hallucination_rate=0.0,
            refusal_precision=0.0,
            refusal_recall=0.0,
        ),
    )


@pytest.mark.parametrize(
    ("audit", "answered", "invented"),
    [
        # A self-abstention is a refusal: neither answered nor invented.
        (_audit("The documents do not mention it.", claims=(_claim(False),)), False, False),
        (_audit("", refused=True), False, False),
        (_audit("It is 30 days.", claims=(_claim(True),)), True, False),
        (_audit("It is 30 days.", claims=(_claim(False),)), True, True),
        # Shipped with no usable claim: the judge could not confirm it.
        (_audit("It is 30 days.", claims=()), True, True),
        # Answered but not judged: no invented outcome, the question leaves metric 3.
        (_audit("It is 30 days."), True, None),
    ],
)
def test_outcomes(
    monkeypatch: pytest.MonkeyPatch, audit: PerExampleAudit, answered: bool, invented: bool | None
) -> None:
    o = _module(monkeypatch).outcome(audit)
    assert (o["answered"], o["invented"]) == (answered, invented)


def test_the_readings_are_the_registered_sentences(monkeypatch: pytest.MonkeyPatch) -> None:
    m = _module(monkeypatch)

    def row(v: int, n: int, p: float) -> dict[str, object]:
        return {"verity": {"k": v}, "naive": {"k": n}, "p_holm": p, "n_paired": 33}

    prereg = (SCRIPT.parents[1] / m.PREREGISTRATION).read_text(encoding="utf-8")
    prereg = " ".join(prereg.split()).lower()
    for sentence in (
        m.reading("unanswerable_answered", row(3, 20, 0.001)),
        m.reading("answerable_answered", row(50, 62, 0.01)),
    ):
        assert sentence.rstrip(".").lower() in prereg
    assert m.reading("unanswerable_answered", row(3, 5, 0.6)).startswith(
        "On 33 unanswerable questions the checks show no effect"
    )
    assert m.reading("invented_answer", row(1, 9, 0.02)) == "Verity invents fewer answers."
    assert m.reading("answerable_answered", row(55, 62, 0.2)).startswith("No cost")
