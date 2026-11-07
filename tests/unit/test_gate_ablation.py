"""Gate-by-gate replay on stored drafts: right and wrong refusals per check. No model call."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "gate_ablation.py"


def _module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    spec = importlib.util.spec_from_file_location("gate_ablation", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "gate_ablation", module)
    spec.loader.exec_module(module)
    return module


def test_the_replay_agrees_with_the_agent_on_every_answered_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every record Verity shipped passed every check; the replay must say so."""
    ga = _module(monkeypatch)
    from verity.config import Settings

    runs = [p for ps in ga.VERITY_RUNS.values() for p in ps]
    assert ga.check_replay(runs, Settings(_env_file=None)) == []  # type: ignore[call-arg]


def test_irrelevant_citation_alone_on_the_naive_dev_drafts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Counted by hand from the e644a98 naive drafts: three answerable drafts cite a chunk
    the reranker scores below 0 (q-001 -0.62, q-023 -2.45, q-063 -2.29), no
    unanswerable draft does."""
    ga = _module(monkeypatch)
    from verity.config import Settings

    report = ga.replay(ga.NAIVE_RUNS["dev"], Settings(_env_file=None))  # type: ignore[call-arg]
    gate = report["alone"]["irrelevant_citation"]
    assert (gate["right"], gate["wrong"], gate["wrong_ids"]) == (0, 3, ["q-001", "q-023", "q-063"])
    assert report["n_drafts"] == 28 and report["n_answerable"] == 24


def test_first_failing_cause_on_test(monkeypatch: pytest.MonkeyPatch) -> None:
    ga = _module(monkeypatch)
    causes = ga.first_failing(ga.VERITY_RUNS["test"])["by_cause"]
    assert causes["irrelevant_citation"] == {"right": 0, "wrong": 4}
    assert causes["no_citation"] == {"right": 6, "wrong": 0}


def test_committed_outputs_are_current(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _module(monkeypatch).main(["--check"]) == 0
