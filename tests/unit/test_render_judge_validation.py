"""The judge-validation blocks of docs/results.md and the judge label come from the artefacts."""

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


def _prov() -> Any:
    return _load("render_judge_validation").Provenance(
        real_by_run={"eval_6ec544d_dev.json": 16, "eval_969e1d5_test.json": 24},
        perturbed=20,
        prompt_version="0.1.0",
        prompt_sha256="2c5f8eca49baa80fc89288c8ceafdc832a5ab3d442dbe601d5e065be40bb1bde",
        exposed_to_injection=4,
    )


def test_the_committed_results_doc_matches_the_agreement_artefact() -> None:
    assert _load("render_judge_validation").main(["--check"]) == 0


def test_blind_figures_are_rendered_with_n_and_the_upper_bound_note() -> None:
    block = _load("render_judge_validation").render_block(_agreement(), "claude-opus-5-5", _prov())
    assert "| Cohen's κ, all claims | 0.71 | 60 claims" in block
    assert "53/60" in block and "16/20" in block and "**upper bound**" in block
    assert "date 5/5 · number 3/5" in block
    assert "sensitivity is an upper bound." in block


def test_kappa_on_a_class_of_fewer_than_five_is_marked_fragile() -> None:
    mod = _load("render_judge_validation")
    block = mod.render_block(_agreement(human_unsup_real=1), "m", _prov())
    real_row = next(line for line in block.splitlines() if "real claims only" in line)
    assert "**fragile**: only 1 of 40 claims is" in real_row
    mixed_row = next(line for line in block.splitlines() if "all claims" in line)
    assert "fragile" not in mixed_row


def _v2(n_judged: int = 60) -> dict[str, Any]:
    result = _agreement(judge_unsup_human_sup=2)
    result["agreement_mixed"]["cohen_kappa"] = 0.85
    result.update(
        {
            "judge_model": "claude-opus-5-5",
            "prompt_version": "0.2.0",
            "prompt_sha256": "ab" * 32,
            "cost_usd": 1.234,
            "n_items": 60,
            "n_judged": n_judged,
            "complete": n_judged == 60,
            "citation_named": {"k": 55, "n": 62},
        }
    )
    result["n_unjudged"], result["n_malformed"] = 60 - n_judged, 0
    same = _agreement()
    same["agreement_mixed"]["cohen_kappa"] = 0.64
    result["v010_on_the_same_claims"] = same
    return result


def _offline(mod: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Provenance and the v0.2.0 artefact from fixtures, not from the repository."""
    monkeypatch.setattr(mod, "provenance", _prov)
    v2 = tmp_path / "v2.json"
    v2.write_text(json.dumps(_v2()))
    monkeypatch.setattr(mod, "V2_AGREEMENT", v2)
    monkeypatch.setattr(mod, "check_current_prompt", lambda result: None)


def test_check_writes_nothing_and_fails_on_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load("render_judge_validation")
    _offline(mod, monkeypatch, tmp_path)
    doc = tmp_path / "results.md"
    doc.write_text(
        "x\n<!-- JUDGE:START -->\nstale\n<!-- JUDGE:END -->\n"
        "<!-- JUDGE-ADJUDICATED:START -->\nstale\n<!-- JUDGE-ADJUDICATED:END -->\n"
        "<!-- JUDGE-V2:START -->\nstale\n<!-- JUDGE-V2:END -->\n"
    )
    before = doc.read_text()
    assert mod.main(["--results-doc", str(doc), "--check"]) == 1
    assert doc.read_text() == before


def test_missing_markers_fail_loudly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load("render_judge_validation")
    _offline(mod, monkeypatch, tmp_path)
    doc = tmp_path / "results.md"
    doc.write_text("no markers\n")
    with pytest.raises(SystemExit, match="markers"):
        mod.main(["--results-doc", str(doc)])


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


def test_the_adjudicated_block_sits_beside_the_blind_figures_and_counts_moves_to_the_judge() -> (
    None
):
    mod = _load("render_judge_validation")
    blind = _agreement()
    adjudicated = _agreement(judge_unsup_human_sup=0, human_unsup_real=3)
    adjudicated["agreement_mixed"]["cohen_kappa"] = 0.93
    adjudicated["adjudication"] = {
        "n_adjudicated": 10,
        "n_changed": 7,
        "n_changed_to_judge": 6,
        "perturbed_judged_true_blind": {
            "n": 7,
            "perturbation stays true": 2,
            "annotator error": 5,
        },
    }
    block = mod.render_adjudicated_block(adjudicated, blind)
    assert "| Cohen's κ, all claims | 0.71 | 0.93 |" in block
    assert "**6 of them to the judge's verdict**" in block
    assert "validate nothing" in block
    assert "(fragile: 3 unsupported)" in block


def test_kappa_is_attached_to_the_judge_prompt_version_it_was_measured_on() -> None:
    block = _load("render_judge_validation").render_block(_agreement(), "m", _prov())
    assert "judge prompt v0.1.0" in block
    assert "`2c5f8eca49ba`" in block
    assert "a changed judge prompt is a new judge and needs its own κ" in block


def test_the_block_says_robustness_to_injection_is_not_measured() -> None:
    block = _load("render_judge_validation").render_block(_agreement(), "m", _prov())
    line = next(x for x in block.splitlines() if "injection" in x)
    assert "not measured" in line
    assert "4 of the 60 claims" in line
    assert "THREAT_MODEL.md#llm01-prompt-injection" in line


def test_real_claims_are_counted_per_run_they_come_from() -> None:
    block = _load("render_judge_validation").render_block(_agreement(), "m", _prov())
    assert "40 real claims (16 from the dev run, 24 from the test run)" in block
    assert "from the test run and" not in block


def test_the_committed_provenance_matches_the_runs_and_the_key() -> None:
    prov = _load("render_judge_validation").provenance()
    assert prov.real_by_run == {"eval_6ec544d_dev.json": 16, "eval_969e1d5_test.json": 24}
    assert prov.perturbed == 20
    assert prov.prompt_version == "0.1.0"
    assert prov.exposed_to_injection == 4


def test_a_changed_judge_prompt_refuses_to_render(tmp_path: Path) -> None:
    """κ measured with one prompt must not be published under another."""
    mod = _load("render_judge_validation")
    measured = {"prompt_sha256": mod._sha256(mod.JUDGE_PROMPT)}
    mod.check_current_prompt(measured)
    changed = tmp_path / "judge_faithfulness.prompt"
    changed.write_text(mod.JUDGE_PROMPT.read_text() + "\n# any edit makes a new judge\n")
    with pytest.raises(SystemExit, match="changed since"):
        mod.check_current_prompt(measured, prompt=changed)


def test_the_v010_kappa_is_labelled_from_the_prompt_the_runs_recorded(tmp_path: Path) -> None:
    mod = _load("render_judge_validation")
    with pytest.raises(SystemExit, match="not the judge prompt the runs recorded"):
        mod.provenance(prompt=mod.JUDGE_PROMPT)


def test_the_v2_block_compares_with_v010_and_names_its_protocol() -> None:
    block = _load("render_judge_validation").render_v2_block(_v2(), _agreement())
    assert "| Cohen's κ, all claims | 0.64 | 0.85 |" in block
    assert "On all 60 claims v0.1.0 has κ 0.71" in block
    assert "judge prompt v0.2.0 (sha256 `abababababab`)" in block
    assert "for $1.2340, on all 60 claims." in block
    assert "mixes the prompt and the protocol" in block
    assert "55 of the 62 claims" in block and "not validated against a human" in block


def test_an_incomplete_rejudge_is_published_as_incomplete() -> None:
    block = _load("render_judge_validation").render_v2_block(_v2(n_judged=52), _agreement())
    assert "**52 of 60 claims only: the budget stopped the rejudge before 8**" in block
    assert "not a random sample" in block
    assert "incomplete" in block


def test_the_injection_count_does_not_depend_on_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert _load("render_judge_validation").provenance().exposed_to_injection == 4
