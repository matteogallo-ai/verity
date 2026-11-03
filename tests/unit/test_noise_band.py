"""The run-to-run noise band: only comparable runs, and the committed block is current."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "noise_band.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("noise_band", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["noise_band"] = module
    spec.loader.exec_module(module)
    return module


def _run(created: str, faithfulness: float) -> dict[str, Any]:
    scorecard = {
        "git_sha": "abc1234",
        "dataset": "dev",
        "n_examples": 3,
        "created_at": created,
        "answer": {
            "faithfulness": faithfulness,
            "citation_accuracy": 0.9,
            "hallucination_rate": 0.0,
            "refusal_precision": 0.5,
            "refusal_recall": 1.0,
        },
        "retrieval": {"ndcg_at_k": 0.9, "recall_at_k": 1.0, "precision_at_k": 0.2},
        "latency": {"p50_ms": 1.0, "p95_ms": 2.0, "p99_ms": 3.0},
        "cold_start_ms": 4.0,
        "cost_per_query_usd": 0.01,
    }
    return {
        "agent_model": "a",
        "judge_model": "j",
        "embedding_model": "e",
        "dataset_sha256": "d",
        "prompt_sha256": {"synthesize": "p"},
        "confidence_threshold": 0.55,
        "scorecard": scorecard,
    }


def test_the_range_is_max_minus_min_over_the_runs() -> None:
    summary = _module().summarise(
        [("r1", _run("t1", 0.9)), ("r2", _run("t2", 1.0)), ("r3", _run("t3", 0.95))]
    )
    faith = next(m for m in summary["metrics"] if m["metric"].startswith("Faithfulness"))
    assert faith["values"] == [0.9, 1.0, 0.95]
    assert (faith["min"], faith["max"]) == (0.9, 1.0)
    assert faith["range"] == pytest.approx(0.1)
    assert summary["n_runs"] == 3


@pytest.mark.parametrize(
    ("field", "value"),
    [("prompt_sha256", {"synthesize": "other"}), ("judge_model", "other"), ("dataset_sha256", "x")],
)
def test_runs_that_differ_in_anything_but_sampling_are_refused(field: str, value: Any) -> None:
    other = _run("t2", 1.0)
    other[field] = value
    with pytest.raises(SystemExit, match=field):
        _module().summarise([("r1", _run("t1", 0.9)), ("r2", other)])


def test_a_run_from_another_commit_is_refused() -> None:
    other = _run("t2", 1.0)
    other["scorecard"]["git_sha"] = "fff0000"
    with pytest.raises(SystemExit, match="git_sha"):
        _module().summarise([("r1", _run("t1", 0.9)), ("r2", other)])


def test_the_same_run_listed_twice_is_refused() -> None:
    run = _run("t1", 0.9)
    with pytest.raises(SystemExit, match="twice"):
        _module().summarise([("r1", run), ("r1-copy", copy.deepcopy(run))])


def test_one_run_is_not_a_noise_band() -> None:
    with pytest.raises(SystemExit, match="at least two"):
        _module().summarise([("r1", _run("t1", 0.9))])


def test_the_committed_summary_and_doc_block_are_current() -> None:
    assert _module().main(["--check"]) == 0


def test_the_reruns_are_pinned() -> None:
    noise = SCRIPT.parents[1] / "datasets" / "eval" / "runs" / "noise"
    pinned = dict(
        reversed(line.split()) for line in (noise / "SHA256SUMS").read_text().splitlines() if line
    )
    assert set(pinned) == {p.name for p in noise.glob("eval_*.json")}
    for name, digest in pinned.items():
        assert hashlib.sha256((noise / name).read_bytes()).hexdigest() == digest, name


def test_the_rendered_block_says_n_is_a_minimum() -> None:
    module = _module()
    block = module.render_block(json.loads(module.SUMMARY.read_text()))
    assert "n=3 runs is a minimum" in block
