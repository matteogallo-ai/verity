"""The completion of the naive-baseline comparison (amendment of 2026-11-05)."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load() -> ModuleType:
    path = REPO_ROOT / "scripts" / "build_completion_subset.py"
    spec = importlib.util.spec_from_file_location("build_completion_subset", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


subset = _load()


def _rows(n: int) -> list[dict[str, object]]:
    return [{"id": f"q-{i:03d}", "answerable": i % 2 == 0} for i in range(n)]


def test_each_arm_gets_only_the_questions_it_did_not_complete() -> None:
    rows = _rows(10)
    verity, naive = subset.subsets(rows, {"q-000", "q-001"}, {"q-000", "q-001", "q-002"})
    assert {r["id"] for r in verity} == {f"q-{i:03d}" for i in range(2, 10)}
    assert {r["id"] for r in naive} == {f"q-{i:03d}" for i in range(3, 10)}
    assert [r for r in verity if r["id"] != "q-002"] == naive  # same order, same rows


def test_the_order_is_drawn_by_the_seed_not_the_id_order() -> None:
    rows = _rows(40)
    first, _ = subset.subsets(rows, set(), set())
    again, _ = subset.subsets(list(reversed(rows)), set(), set())
    other, _ = subset.subsets(rows, set(), set(), seed=subset.SEED + 1)
    assert [r["id"] for r in first] == [r["id"] for r in again]
    assert [r["id"] for r in first] != sorted(r["id"] for r in first)
    assert [r["id"] for r in first] != [r["id"] for r in other]


def test_the_committed_files_are_current() -> None:
    assert subset.main(["--check"]) == 0


def test_the_committed_rows_are_dev_rows_unchanged() -> None:
    dev = {json.loads(line)["id"]: line for line in subset.DEV.read_text().splitlines()}
    for path in (subset.VERITY_OUT, subset.NAIVE_OUT):
        for line in path.read_text().splitlines():
            assert json.loads(line) == json.loads(dev[json.loads(line)["id"]])


def test_the_completion_caps_stay_under_the_amount_asked_for() -> None:
    script = (REPO_ROOT / "scripts" / "run_naive_baseline.sh").read_text()
    block = script[script.index("completion)") :]
    caps = [float(v) for v in re.findall(r"_CAP_USD=([\d.]+)", block)[:2]]
    # The harness checks the cap after each question: each arm can pass it by its
    # dearest question, measured on the runs at the commit whose code this one holds.
    overshoot = 0.0
    for run in (subset.VERITY_RUN, subset.NAIVE_RUN):
        rows = json.loads(run.read_text())["per_audit"]
        overshoot += max(
            r["usage"]["cost_usd"] + ((r.get("judge_usage") or {}).get("cost_usd") or 0.0)
            for r in rows
        )
    assert sum(caps) + overshoot <= 1.15
