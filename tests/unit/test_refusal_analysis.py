"""``scripts/refusal_analysis.py``: the threshold is chosen on dev only and the test
split is measured with it; calibrators are fitted on dev and scored on test."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from verity.eval.calibration import ScoredExample
from verity.types import RefusalReason

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "refusal_analysis.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("refusal_analysis", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ex(answerable: bool, score: float) -> ScoredExample:
    return ScoredExample(answerable, score, RefusalReason.BELOW_THRESHOLD if score < 0.5 else None)


# Unanswerable questions score at most 0.3, answerable ones at least 0.4: on dev the
# best threshold is the lowest one that refuses every unanswerable question.
_DEV = [_ex(False, 0.1), _ex(False, 0.3), _ex(True, 0.4), _ex(True, 0.8), _ex(True, 0.9)]
# On test an answerable question scores 0.2: it must be refused at the dev threshold,
# not rescued by re-tuning on test.
_TEST = [_ex(False, 0.25), _ex(True, 0.2), _ex(True, 0.7), _ex(True, 0.95)]


def test_threshold_comes_from_dev_and_is_applied_unchanged_to_test(tmp_path: Path) -> None:
    mod = _module()
    report = mod.analyse(_DEV, _TEST)
    assert report["dev"]["chosen"]["threshold"] == pytest.approx(0.35)
    at_test = report["test"]["at_chosen_threshold"]
    assert at_test["threshold"] == pytest.approx(0.35)
    assert (at_test["tp"], at_test["fp"]) == (1, 1)
    assert set(report["test"]["calibration"]) == {"raw", "isotonic", "platt"}
    assert report["test"]["calibration"]["raw"]["n"] == 4

    png = tmp_path / "reliability.png"
    mod.plot(report, png)
    assert png.read_bytes().startswith(b"\x89PNG")


def test_dev_only_report_has_no_test_section() -> None:
    assert "test" not in _module().analyse(_DEV, None)


def test_the_fallback_rule_is_applied_and_named_when_precision_is_out_of_reach() -> None:
    # An answerable question scores below every unanswerable one: no threshold that
    # refuses an unanswerable question avoids refusing it too.
    dev = [_ex(True, 0.05), _ex(False, 0.1), _ex(False, 0.3), _ex(True, 0.9)]
    report = _module().analyse(dev, None)
    assert report["rule"].startswith("fallback")
    assert report["dev"]["chosen"]["threshold"] == pytest.approx(0.35)


def test_single_class_dev_scores_skip_recalibration_instead_of_crashing(tmp_path: Path) -> None:
    # Every unanswerable dev question was refused by a deterministic check (no score).
    gate = RefusalReason.NO_CITATION
    dev = [ScoredExample(False, 0.0, gate), _ex(True, 0.6), _ex(True, 0.9)]
    report = _module().analyse(dev, _TEST)
    assert report["test"]["calibration"]["recalibration"].startswith("not fitted")
    assert "isotonic" not in report["test"]["calibration"]


def test_a_run_without_scored_examples_still_plots(tmp_path: Path) -> None:
    gate = RefusalReason.NO_CITATION
    report = _module().analyse([ScoredExample(False, 0.0, gate)], None)
    assert report["dev"]["calibration"]["ece"] is None
    _module().plot(report, tmp_path / "r.png")
    assert (tmp_path / "r.png").is_file()


_REPO = SCRIPT.parents[1]
_LIVE = _REPO / "datasets" / "eval" / "runs" / "live"
_ARGS = [
    "--dev",
    str(_LIVE / "eval_6ec544d_dev.json"),
    "--test",
    str(_LIVE / "eval_969e1d5_test.json"),
    "--out-json",
    str(_REPO / "scorecards" / "refusal-analysis.json"),
]


def test_committed_analysis_is_what_the_live_artefacts_give() -> None:
    assert _module().main([*_ARGS, "--check"]) == 0


def test_check_fails_on_a_hand_edited_analysis_and_writes_nothing(tmp_path: Path) -> None:
    edited = tmp_path / "analysis.json"
    text = (_REPO / "scorecards" / "refusal-analysis.json").read_text(encoding="utf-8")
    edited.write_text(text.replace('"threshold": 0.0', '"threshold": 0.55', 1), encoding="utf-8")
    before = edited.read_text(encoding="utf-8")
    assert _module().main([*_ARGS[:-1], str(edited), "--check"]) == 1
    assert edited.read_text(encoding="utf-8") == before
