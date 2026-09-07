"""Physical path separation of live and stub audit artefacts.

The S7 pre-tag audit exposed this class of accident: the CLI wrote both
stub verification runs and live runs of record to the same
``datasets/eval/runs/eval_<sha>_questions.json`` path. A subsequent stub
run (from CI, from ``verity eval run --ci``, from a maintainer's manual
verification) silently destroyed the live run's ``per_audit`` audit trail.

The primary defence is **path separation**: live runs (any run whose agent
OR judge is a real LLM) auto-nest into a ``live/`` subdirectory of the
runs_dir; stub runs stay at the plain runs_dir. This module locks that
contract with three orthogonal proofs :

1. **Stub runs never write to the live subtree.** Even with the safeguard
   disabled, ``verity eval run --ci`` cannot create a file inside
   ``runs_dir/live/``.
2. **Live-mode runs auto-route to the live subtree.** No operator opt-in,
   no ``--live`` flag — the path separation is a property of the run's
   provenance.
3. **The belt-and-braces safeguard fires** when someone bypasses the path
   separation and asks a stub-provenance ``EvalRun`` to overwrite an
   existing live artefact. Preserves the audit trail even if a future
   refactor breaks the path split.

Zero LLM cost — stub agent + stub judge on the real 16-question dataset;
the live-mode assertion is exercised via the ``StubOverwritesLiveError``
directly on ``FileRunStore`` without any LLM call.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from verity.cli import app, resolve_runs_dir
from verity.eval.run_store import FileRunStore, StubOverwritesLiveError
from verity.types import (
    AnswerMetrics,
    EvalRun,
    LatencyMetrics,
    RetrievalMetrics,
    Scorecard,
)

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", _ANSI_RE.sub("", text)).strip()


def _make_run(*, agent_model: str, judge_model: str, git_sha: str = "test0000") -> EvalRun:
    """Minimal EvalRun with the provenance labels that matter for the safeguard."""
    return EvalRun(
        scorecard=Scorecard(
            git_sha=git_sha,
            created_at=datetime(2026, 9, 7, 12, 0, tzinfo=UTC),
            dataset="questions",
            n_examples=16,
            retrieval=RetrievalMetrics(precision_at_k=0.15, recall_at_k=1.0, ndcg_at_k=0.9, k=8),
            answer=AnswerMetrics(
                faithfulness=1.0,
                citation_accuracy=1.0,
                hallucination_rate=0.0,
                refusal_precision=1.0,
                refusal_recall=1.0,
            ),
            latency=LatencyMetrics(p50_ms=80.0, p95_ms=200.0, p99_ms=400.0),
            cost_per_query_usd=0.0,
        ),
        embedding_model="bge-small",
        judge_model=judge_model,
        agent_model=agent_model,
    )


# --------------------------------------------------------------------------------------
# 0. Truth-table proof for the routing rule on all 4 (judge, agent) combos.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("judge_mode", "agent_stub", "expected_subdir", "why"),
    [
        pytest.param(
            "stub",
            True,
            None,
            "stub judge + stub agent → $0 → plain runs_dir (freely overwritable)",
            id="stub_judge__stub_agent",
        ),
        pytest.param(
            "stub",
            False,
            "live",
            "stub judge + LIVE agent (--no-agent-stub) → agent spends → live/",
            id="stub_judge__live_agent",
        ),
        pytest.param(
            "live",
            True,
            "live",
            "LIVE judge (--judge live) + stub agent → judge spends → live/",
            id="live_judge__stub_agent",
        ),
        pytest.param(
            "live",
            False,
            "live",
            "LIVE judge + LIVE agent → both spend → live/",
            id="live_judge__live_agent",
        ),
    ],
)
def test_resolve_runs_dir_covers_every_paid_combo(
    tmp_path: Path,
    judge_mode: str,
    agent_stub: bool,
    expected_subdir: str | None,
    why: str,
) -> None:
    """The load-bearing invariant: **any run that spends a single dollar
    persists under ``live/``**. Stub-only runs stay at the plain path where
    CI can freely overwrite. Proven on all 4 (judge_mode, agent_stub)
    combinations reachable via the CLI, including the two hybrid modes
    (stub judge + live agent, live judge + stub agent) — either alone is
    enough to route to ``live/``."""
    base = tmp_path / "runs"
    got = resolve_runs_dir(base, judge_mode=judge_mode, agent_stub=agent_stub)
    if expected_subdir is None:
        assert got == base, f"{why}: expected plain path {base}, got {got}"
    else:
        assert got == base / expected_subdir, f"{why}: expected {base / expected_subdir}, got {got}"


def test_no_paid_combo_is_ever_routed_to_the_plain_path() -> None:
    """Belt-and-braces on the truth table: for every (judge, agent) combo
    where ANY side is non-stub (any $ spend), the effective path must NOT
    be the plain runs_dir. Formulates the invariant as a universal claim
    ('no paid combo maps to the overwritable path') rather than four
    positive checks — so a future rule refactor that flips a boolean
    silently still fails this test."""
    base = Path("/tmp/_runs")
    for judge_mode in ("stub", "live"):
        for agent_stub in (True, False):
            spends = (judge_mode == "live") or (not agent_stub)
            got = resolve_runs_dir(base, judge_mode=judge_mode, agent_stub=agent_stub)
            if spends:
                assert got != base, (
                    f"paid combo judge={judge_mode!r} agent_stub={agent_stub} "
                    "routed to the overwritable plain path — audit-loss risk"
                )
                assert got == base / "live"
            else:
                assert got == base


# --------------------------------------------------------------------------------------
# 1. Stub runs stay at plain runs_dir; nothing ever lands under live/.
# --------------------------------------------------------------------------------------


def test_stub_run_writes_at_plain_runs_dir_never_under_live_subtree(tmp_path: Path) -> None:
    """``verity eval run --ci`` writes to ``<runs_dir>/eval_<sha>_questions.json``
    and never creates or touches anything under ``<runs_dir>/live/``. This is
    the load-bearing invariant — a stub verification run can freely overwrite
    its own artefacts in CI without ever putting a live audit trail at risk."""
    runs_dir = tmp_path / "runs"
    result = runner.invoke(app, ["eval", "run", "--ci", "--runs-dir", str(runs_dir)])
    assert result.exit_code == 0, f"stub run failed: {result.output}"

    # Stub file exists at the plain path.
    stub_files = list(runs_dir.glob("eval_*_questions.json"))
    assert len(stub_files) == 1, (
        f"expected one stub artefact at the plain runs_dir; got {stub_files}"
    )
    # Live subtree is absent (or empty) — nothing leaked into it.
    live_subdir = runs_dir / "live"
    if live_subdir.exists():
        leaked = list(live_subdir.glob("*.json"))
        assert not leaked, f"stub run leaked artefacts into the live subtree: {leaked}"


# --------------------------------------------------------------------------------------
# 2. Live-provenance runs auto-route to the live/ subtree.
# --------------------------------------------------------------------------------------


def test_live_provenance_run_routes_to_live_subtree(tmp_path: Path) -> None:
    """A run whose agent OR judge is a real LLM (i.e. any non-stub) must
    persist under ``runs_dir/live/``, not the plain runs_dir. Proven by
    directly writing a live-provenance :class:`EvalRun` through
    :class:`FileRunStore` at the ``live/`` subpath and asserting the file
    lands there and only there — no LLM calls involved."""
    runs_dir = tmp_path / "runs"
    live_subdir = runs_dir / "live"

    # The CLI's rule: any non-stub → nested under live/. We simulate exactly
    # that split here to lock the effective-path convention.
    live_store = FileRunStore(live_subdir)
    live_run = _make_run(agent_model="claude-sonnet-4-6", judge_model="claude-sonnet-4-6")
    live_store.save(live_run)

    # Landed under live/ …
    live_files = list(live_subdir.glob("eval_*_questions.json"))
    assert len(live_files) == 1

    # … and did NOT appear at the plain runs_dir.
    if runs_dir.exists():
        plain_leak = [p for p in runs_dir.glob("eval_*_questions.json") if p.parent == runs_dir]
        assert not plain_leak, f"live-provenance run leaked into the stub tree: {plain_leak}"


# --------------------------------------------------------------------------------------
# 3. Belt-and-braces: safeguard fires if someone bypasses the path split.
# --------------------------------------------------------------------------------------


def test_stub_cannot_overwrite_live_artefact_at_the_same_path(tmp_path: Path) -> None:
    """Even if a future refactor breaks the path separation, the ``FileRunStore``
    safeguard must refuse to let a stub-provenance run clobber a live
    artefact at the same ``(sha, dataset)`` path. Bypass requires an explicit
    ``allow_overwrite=True`` — the operator has to say the audit-destroying
    action out loud."""
    store = FileRunStore(tmp_path)
    live_run = _make_run(
        agent_model="claude-sonnet-4-6",
        judge_model="claude-sonnet-4-6",
        git_sha="livesha",
    )
    store.save(live_run)
    live_path = store._path_for(live_run.scorecard.git_sha, live_run.scorecard.dataset)
    original_bytes = live_path.read_bytes()

    stub_run = _make_run(
        agent_model="stub-agent",
        judge_model="stub-judge-v1",
        git_sha="livesha",  # same sha → same path
    )
    with pytest.raises(StubOverwritesLiveError):
        store.save(stub_run)

    # The live artefact on disk is byte-identical after the blocked attempt.
    assert live_path.read_bytes() == original_bytes, (
        "safeguard fired but the live file was still modified — data-loss regression"
    )


def test_stub_can_freely_write_its_own_artefact_when_no_live_conflict(
    tmp_path: Path,
) -> None:
    """The safeguard must NOT punish routine stub runs. When the path is
    empty (fresh CI) or already contains a stub-provenance file, the save
    succeeds without ceremony."""
    store = FileRunStore(tmp_path)

    stub_run = _make_run(agent_model="stub-agent", judge_model="stub-judge-v1", git_sha="stubsha")
    store.save(stub_run)  # fresh path
    store.save(stub_run)  # overwriting an existing stub file: also fine
    # Two saves, one file: proves overwrite works when both sides are stub.
    files = list(tmp_path.glob("eval_*_questions.json"))
    assert len(files) == 1


def test_explicit_allow_overwrite_bypasses_the_safeguard(tmp_path: Path) -> None:
    """The escape hatch: ``allow_overwrite=True`` lets a stub run clobber a
    live artefact intentionally (e.g. an operator explicitly resetting an
    old run). Kept behind an explicit opt-in — accidental clobbers stay
    blocked; deliberate ones are one keyword away."""
    store = FileRunStore(tmp_path)
    live_run = _make_run(
        agent_model="claude-sonnet-4-6",
        judge_model="claude-sonnet-4-6",
        git_sha="livesha",
    )
    store.save(live_run)
    stub_run = _make_run(agent_model="stub-agent", judge_model="stub-judge-v1", git_sha="livesha")
    store.save(stub_run, allow_overwrite=True)  # must NOT raise
    on_disk = json.loads(store._path_for("livesha", "questions").read_text(encoding="utf-8"))
    assert on_disk["agent_model"] == "stub-agent"


# --------------------------------------------------------------------------------------
# 4. CLI-level assertion: the exit code and message for the safeguard.
# --------------------------------------------------------------------------------------


def test_cli_exit_code_five_when_stub_run_hits_live_artefact(tmp_path: Path) -> None:
    """End-to-end: a stub CLI run targeting a path that already contains a
    live artefact exits 5 with an actionable message pointing at
    ``allow_overwrite=True``. Confirms the safeguard is wired all the way
    through Typer, not just at the store layer."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    # Seed a live artefact at the exact SHA the stub run will use. We read
    # the SHA from a first invocation via a fresh runs_dir so the test is
    # tolerant of whatever ``current_git_sha()`` returns in this checkout.
    scratch = tmp_path / "scratch"
    first = runner.invoke(app, ["eval", "run", "--ci", "--runs-dir", str(scratch)])
    assert first.exit_code == 0
    stub_files = list(scratch.glob("eval_*_questions.json"))
    assert len(stub_files) == 1
    seed_name = stub_files[0].name  # e.g. eval_d0ca1d5_questions.json

    # Now write a live-provenance file at that exact name in ``runs_dir``.
    live_run = _make_run(
        agent_model="claude-sonnet-4-6",
        judge_model="claude-sonnet-4-6",
        git_sha=seed_name.split("_")[1],  # extract SHA from filename
    )
    FileRunStore(runs_dir).save(live_run)

    # Stub CLI run at the same path → blocked, exit 5.
    result = runner.invoke(app, ["eval", "run", "--ci", "--runs-dir", str(runs_dir)])
    assert result.exit_code == 5, (
        f"expected exit 5 for stub-over-live guard; got {result.exit_code}. "
        f"output={result.output!r}"
    )
    combined = _clean(result.output or "")
    assert "refusing to overwrite" in combined
    assert "allow_overwrite" in combined
