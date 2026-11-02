"""Published live run artefacts are immutable.

``datasets/eval/runs/live/SHA256SUMS`` pins every run of record. A new run adds a
line; no line ever changes and no listed file may change. A metric recomputed under a
newer definition goes into ``<artefact>.recomputed.json``, written by
``scripts/recompute_live_aggregates.py``, which must stay current.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LIVE = REPO_ROOT / "datasets" / "eval" / "runs" / "live"
SCRIPT = REPO_ROOT / "scripts" / "recompute_live_aggregates.py"


def _pinned() -> dict[str, str]:
    lines = (LIVE / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    return {name: digest for digest, name in (line.split() for line in lines if line.strip())}


def _artefacts() -> list[Path]:
    return sorted(p for p in LIVE.glob("eval_*.json") if not p.name.endswith(".recomputed.json"))


def test_every_live_artefact_is_pinned() -> None:
    assert {p.name for p in _artefacts()} == set(_pinned())


@pytest.mark.parametrize("name", sorted(_pinned()))
def test_a_pinned_artefact_never_changes(name: str) -> None:
    digest = hashlib.sha256((LIVE / name).read_bytes()).hexdigest()
    assert digest == _pinned()[name], f"{name} was modified; published runs are immutable"


def test_recomputed_aggregates_are_current() -> None:
    spec = importlib.util.spec_from_file_location("recompute_live_aggregates", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--check"]) == 0
