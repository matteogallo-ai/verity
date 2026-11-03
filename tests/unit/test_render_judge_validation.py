"""The README judge-validation block and the judge label come from the artefacts."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _agreement(judge_unsup_human_sup: int = 6, human_unsup_real: int = 1) -> dict[str, Any]:
    def p(k: int, n: int) -> dict[str, Any]:
        return {"k": k, "n": n, "value": k / n, "ci95_wilson": [0.1, 0.9]}

    def agreement(n: int, ss: int, su: int, us: int, uu: int, kappa: float) -> dict[str, Any]:
        return {
            "n_claims": n,
            "cohen_kappa": kappa,
            "observed": p(ss + uu, n),
            "confusion": {
                "judge_supported/human_supported": ss,
                "judge_supported/human_unsupported": su,
                "judge_unsupported/human_supported": us,
                "judge_unsupported/human_unsupported": uu,
            },
        }

    return {
        "agreement_mixed": agreement(60, 40, 1, judge_unsup_human_sup, 13, 0.71),
        "agreement_real_only": agreement(40, 37, 0, 2, human_unsup_real, 0.48),
        "specificity_real": p(37, 39),
        "sensitivity": {
            "judge_detected": p(16, 20),
            "judge_detected_among_human_unsupported": p(12, 13),
            "human_unsupported": p(13, 20),
            "by_kind": {"date": p(5, 5), "number": p(3, 5)},
        },
        "notes": ["sensitivity is an upper bound"],
    }


def test_the_committed_readme_matches_the_agreement_artefact() -> None:
    assert _load("render_judge_validation").main(["--check"]) == 0


def test_blind_figures_are_rendered_with_n_and_the_upper_bound_note() -> None:
    block = _load("render_judge_validation").render_block(_agreement(), "claude-opus-5-5")
    assert "| Cohen's κ, all claims | 0.71 | 60 claims" in block
    assert "53/60" in block and "16/20" in block and "**upper bound**" in block
    assert "date 5/5 · number 3/5" in block
    assert "sensitivity is an upper bound." in block


def test_kappa_on_a_class_of_fewer_than_five_is_marked_fragile() -> None:
    mod = _load("render_judge_validation")
    block = mod.render_block(_agreement(human_unsup_real=1), "m")
    real_row = next(line for line in block.splitlines() if "real claims only" in line)
    assert "**fragile**: only 1 of 40 claims is" in real_row
    mixed_row = next(line for line in block.splitlines() if "all claims" in line)
    assert "fragile" not in mixed_row


def test_check_writes_nothing_and_fails_on_drift(tmp_path: Path) -> None:
    mod = _load("render_judge_validation")
    readme = tmp_path / "README.md"
    readme.write_text("x\n<!-- JUDGE:START -->\nstale\n<!-- JUDGE:END -->\n")
    before = readme.read_text()
    assert mod.main(["--readme", str(readme), "--check"]) == 1
    assert readme.read_text() == before


def test_missing_markers_fail_loudly(tmp_path: Path) -> None:
    mod = _load("render_judge_validation")
    readme = tmp_path / "README.md"
    readme.write_text("no markers\n")
    with pytest.raises(SystemExit, match="markers"):
        mod.main(["--readme", str(readme)])


def _validation(tmp_path: Path, model: str) -> Path:
    (tmp_path / "agreement.json").write_text(json.dumps(_agreement()))
    (tmp_path / "perturbed_verdicts.json").write_text(json.dumps({"judge_model": model}))
    return tmp_path


def test_judge_label_names_kappa_only_for_the_validated_model(tmp_path: Path) -> None:
    gen = _load("generate_headline_results")
    folder = _validation(tmp_path, "claude-opus-5-5")
    assert gen.judge_label("claude-opus-5-5", folder) == "judge κ 0.71 vs blind human (n=60)"
    assert gen.judge_label("gpt-4.1", folder) == "judge unvalidated"
    assert gen.judge_label("claude-opus-5-5", tmp_path / "absent") == "judge unvalidated"
