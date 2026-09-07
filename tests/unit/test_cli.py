"""CLI surface is stable and every command runs (scaffold bodies exit 0)."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from verity import __version__
from verity.cli import MISSING_KEY_MESSAGE, app
from verity.config import get_settings
from verity.eval import FileRunStore
from verity.types import (
    AnswerMetrics,
    EvalRun,
    LatencyMetrics,
    RetrievalMetrics,
    Scorecard,
)

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _clean(output: str) -> str:
    """Strip ANSI escapes and collapse whitespace so wrapping / colouring can't
    break substring assertions on the terminal output."""
    without_ansi = _ANSI_RE.sub("", output)
    return re.sub(r"\s+", " ", without_ansi).strip()


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_eval_run_ci_exits_clean(tmp_path: Path) -> None:
    """Hermetic on ``--runs-dir tmp_path`` so the test never writes to the real
    ``datasets/eval/runs/`` tree. The S7 pre-tag audit exposed that the two
    prior tests running without ``--runs-dir`` would clobber whatever sat at
    the default path — including a committed live-run audit artefact.
    """
    result = runner.invoke(app, ["eval", "run", "--ci", "--runs-dir", str(tmp_path)])
    assert result.exit_code == 0


def test_eval_run_report_only_floors_writes_json_and_exits_zero(tmp_path: Path) -> None:
    """One-shot LIVE-run invariant, exercised on the stub:

    ``--report-only-floors`` + an impossible floor must
      (a) still write the scorecard JSON to disk (the artefact is committed
          regardless — a floor breach must never lose the run of record), AND
      (b) exit 0 (a breach on a run of record must NOT invite a re-spend).

    The stub verifies the wiring — the live run inherits the same semantics."""
    scorecard = tmp_path / "live.json"
    runs_dir = tmp_path / "runs"
    result = runner.invoke(
        app,
        [
            "eval",
            "run",
            "--ci",  # judge stub + agent stub — zero cost
            "--runs-dir",
            str(runs_dir),
            "--scorecard-json",
            str(scorecard),
            "--report-only-floors",
            "--floor-ndcg",
            "0.999",  # impossible on this dataset — will breach
        ],
    )
    assert result.exit_code == 0, (
        f"report-only floors must not exit non-zero; got {result.exit_code}. "
        f"output={result.output!r}"
    )
    # JSON was written despite the breach — the load-bearing invariant.
    assert scorecard.is_file(), "scorecard JSON must be written before floor evaluation"
    combined = _clean(result.output or "")
    assert "below floor" in combined, (
        "the operator must see the breach in the log — report-only ≠ silent"
    )


def test_eval_run_fatal_floors_exit_two_on_stub(tmp_path: Path) -> None:
    """Same impossible floor WITHOUT --report-only-floors → exit 2 (CI behaviour).

    Hermetic on ``--runs-dir tmp_path`` — see the note on
    ``test_eval_run_ci_exits_clean`` above."""
    result = runner.invoke(
        app,
        [
            "eval",
            "run",
            "--ci",
            "--runs-dir",
            str(tmp_path),
            "--floor-ndcg",
            "0.999",
        ],
    )
    assert result.exit_code == 2, (
        f"CI floor breach must exit 2; got {result.exit_code}. output={result.output!r}"
    )


def _seed_run(
    store: FileRunStore,
    *,
    sha: str,
    when: datetime,
    ndcg: float = 0.90,
    refusal_recall: float = 1.0,
    p95: float = 200.0,
) -> None:
    """Seed a controlled ``EvalRun`` under a hermetic runs directory.

    Every seeded run shares the same ``(dataset, agent_model, judge_model)``
    provenance so the compat filter picks them up as comparable — the tests
    exercise the diff logic, not the compatibility filter (that's covered by a
    dedicated harness test elsewhere).
    """
    store.save(
        EvalRun(
            scorecard=Scorecard(
                git_sha=sha,
                created_at=when,
                dataset="questions",
                n_examples=16,
                retrieval=RetrievalMetrics(
                    precision_at_k=0.15,
                    recall_at_k=1.0,
                    ndcg_at_k=ndcg,
                    k=8,
                ),
                answer=AnswerMetrics(
                    faithfulness=1.0,
                    citation_accuracy=1.0,
                    hallucination_rate=0.0,
                    refusal_precision=1.0,
                    refusal_recall=refusal_recall,
                ),
                latency=LatencyMetrics(p50_ms=80.0, p95_ms=p95, p99_ms=400.0),
                cost_per_query_usd=0.0,
            ),
            embedding_model="bge-small",
            judge_model="stub-judge-v1",
            agent_model="stub-agent",
        )
    )


def test_eval_compare_zero_runs_exits_clean(tmp_path: Path) -> None:
    """Fresh checkout with no history → 'nothing to compare' → exit 0.

    Hermetic on ``--runs-dir tmp_path`` so the result is invariant of the real
    repo's ``datasets/eval/runs/`` contents."""
    result = runner.invoke(app, ["eval", "compare", "--runs-dir", str(tmp_path)])
    assert result.exit_code == 0


def test_eval_compare_single_run_exits_clean(tmp_path: Path) -> None:
    """Exactly one persisted run → 'nothing to compare' → exit 0 (no false alarm)."""
    store = FileRunStore(tmp_path)
    _seed_run(store, sha="aaa1111", when=datetime(2026, 9, 6, tzinfo=UTC))
    result = runner.invoke(app, ["eval", "compare", "--runs-dir", str(tmp_path)])
    assert result.exit_code == 0


def test_eval_compare_identical_runs_exits_clean(tmp_path: Path) -> None:
    """Two byte-identical runs → no metric moved → exit 0."""
    store = FileRunStore(tmp_path)
    _seed_run(store, sha="aaa1111", when=datetime(2026, 9, 5, tzinfo=UTC))
    _seed_run(store, sha="bbb2222", when=datetime(2026, 9, 6, tzinfo=UTC))
    result = runner.invoke(app, ["eval", "compare", "--runs-dir", str(tmp_path)])
    assert result.exit_code == 0


def test_eval_compare_regression_exits_three(tmp_path: Path) -> None:
    """A degraded metric between two comparable runs → exit 3."""
    store = FileRunStore(tmp_path)
    _seed_run(
        store,
        sha="prev0000",
        when=datetime(2026, 9, 5, tzinfo=UTC),
        ndcg=0.90,
        refusal_recall=1.0,
    )
    _seed_run(
        store,
        sha="curr0000",
        when=datetime(2026, 9, 6, tzinfo=UTC),
        ndcg=0.60,  # nDCG dropped by 0.30 → regression
        refusal_recall=1.0,
    )
    result = runner.invoke(app, ["eval", "compare", "--runs-dir", str(tmp_path)])
    assert result.exit_code == 3, (
        f"expected exit 3 on regression, got {result.exit_code}; output={result.output!r}"
    )


def test_eval_compare_skips_incompatible_provenance(tmp_path: Path) -> None:
    """Latest run vs a prior run with different (agent, judge) → skipped, exit 0.

    Guards the compat rule: comparing a live-agent run against a stub-agent run
    would surface deltas that reflect the provenance change, not a regression."""
    store = FileRunStore(tmp_path)
    _seed_run(store, sha="stub0000", when=datetime(2026, 9, 5, tzinfo=UTC))
    # Second run has a *different* agent_model → compat filter must skip it.
    store.save(
        EvalRun(
            scorecard=Scorecard(
                git_sha="live0000",
                created_at=datetime(2026, 9, 6, tzinfo=UTC),
                dataset="questions",
                n_examples=16,
                retrieval=RetrievalMetrics(
                    precision_at_k=0.15, recall_at_k=1.0, ndcg_at_k=0.90, k=8
                ),
                answer=AnswerMetrics(
                    faithfulness=1.0,
                    citation_accuracy=1.0,
                    hallucination_rate=0.0,
                    refusal_precision=1.0,
                    refusal_recall=0.4,  # would be a huge "regression" if compared
                ),
                latency=LatencyMetrics(p50_ms=80.0, p95_ms=200.0, p99_ms=400.0),
                cost_per_query_usd=0.0,
            ),
            embedding_model="bge-small",
            judge_model="stub-judge-v1",
            agent_model="claude-sonnet-4-6",  # different agent
        )
    )
    result = runner.invoke(app, ["eval", "compare", "--runs-dir", str(tmp_path)])
    assert result.exit_code == 0, (
        f"incompatible provenance must not surface a fake regression; "
        f"got exit {result.exit_code}, output={result.output!r}"
    )


def test_ingest_exits_clean(tmp_path: object) -> None:
    result = runner.invoke(app, ["ingest", str(tmp_path)])
    assert result.exit_code == 0


def test_missing_key_message_contains_actionable_hints() -> None:
    """Directly assert on the guard message constant — the source of truth.

    Reading the constant instead of parsing terminal output makes the assertion
    completely independent of Rich's panel wrapping, ANSI colouring, and terminal
    width. This is the primary regression guard.
    """
    assert "VERITY_ANTHROPIC_API_KEY" in MISSING_KEY_MESSAGE
    assert "VERITY_OPENAI_API_KEY" in MISSING_KEY_MESSAGE
    assert "--stub" in MISSING_KEY_MESSAGE


def test_ask_without_key_and_without_stub_errors_actionably(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end CLI check: no key → non-zero exit and the guard message reaches
    the terminal without a Python traceback.

    Robustness measures:
    - ``COLUMNS=200`` prevents Rick's BadParameter panel from wrapping tokens like
      ``VERITY_ANTHROPIC_API_KEY`` across lines;
    - ``TERM=dumb`` neutralises ANSI colouring;
    - :func:`_clean` strips any residual ANSI and collapses whitespace so the
      substring checks are tokenisation-independent.
    """
    monkeypatch.delenv("VERITY_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("VERITY_OPENAI_API_KEY", raising=False)
    get_settings.cache_clear()
    try:
        result = runner.invoke(
            app,
            ["ask", "does the contract permit termination?"],
            env={"COLUMNS": "200", "TERM": "dumb", "NO_COLOR": "1"},
        )
    finally:
        get_settings.cache_clear()

    # Behaviour first: the guard triggered and produced a non-zero exit.
    assert result.exit_code != 0

    combined = _clean((result.output or "") + (str(result.exception) if result.exception else ""))
    assert "VERITY_ANTHROPIC_API_KEY" in combined
    assert "--stub" in combined
    # Explicit "no traceback leaked" check — BadParameter must render as a Typer
    # message, not a raw Python stack.
    assert "Traceback" not in combined
