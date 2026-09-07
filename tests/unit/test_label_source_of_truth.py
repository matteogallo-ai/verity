"""Console-vs-JSON label consistency contract.

The S7 pre-tag audit exposed this gap: the JSON scorecard was renamed to
``hallucination_rate_answerable`` + ``out_of_scope_answered`` (schema v2),
but the Rich console table in :func:`verity.cli._print_scorecard` still
printed the pre-v2 name ``hallucination_rate`` with no companion row. A
reader looking at the console got a different vocabulary than a reader
looking at the JSON — the same run, two different label sets.

These tests LOCK the invariant: every headline label that appears in the
CLI console output must exist as a key in the v2 JSON scorecard schema, and
every user-visible metric in the JSON must also appear in the console table.
There is one source of truth for what a metric is called.

Zero LLM cost — stub agent + stub judge on the real 16-question dataset.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from typer.testing import CliRunner

from verity.cli import app

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _clean(text: str) -> str:
    """Strip ANSI + collapse whitespace so substring checks are terminal-agnostic."""
    return re.sub(r"\s+", " ", _ANSI_RE.sub("", text)).strip()


def _run_stub_and_capture(tmp_path: Path) -> tuple[str, dict]:
    """Run ``verity eval run --ci`` with a stub scorecard, return (console, JSON)."""
    scorecard = tmp_path / "sc.json"
    runs_dir = tmp_path / "runs"
    result = runner.invoke(
        app,
        [
            "eval",
            "run",
            "--ci",
            "--runs-dir",
            str(runs_dir),
            "--scorecard-json",
            str(scorecard),
        ],
        env={"COLUMNS": "200", "TERM": "dumb", "NO_COLOR": "1"},
    )
    assert result.exit_code == 0, f"stub run failed: {result.output}"
    assert scorecard.is_file()
    payload = json.loads(scorecard.read_text(encoding="utf-8"))
    return _clean(result.output), payload


# --------------------------------------------------------------------------------------
# Load-bearing invariant: JSON v3 keys ↔ console labels.
# --------------------------------------------------------------------------------------


class TestConsoleJsonLabelInvariant:
    """Every user-visible metric name + denominator is defined ONCE.
    Console and JSON agree on vocabulary AND on scope/n_denominator."""

    def test_json_is_v3_with_scoped_metrics(self, tmp_path: Path) -> None:
        """Baseline: the JSON we generate today is schema v3 — no drift back
        to a v2/v1 shape under any refactor. Each judge-track metric is a
        ``{value, scope, n_denominator}`` triple so no chiffre can be
        quoted without its denominator."""
        _, payload = _run_stub_and_capture(tmp_path)
        assert payload["schema"] == "verity.scorecard.headline/v3"
        for name in ("faithfulness", "citation_accuracy", "hallucination_rate"):
            entry = payload["answer_quality"][name]
            assert set(entry.keys()) == {"value", "scope", "n_denominator"}, (
                f"v3 answer_quality[{name!r}] must be a {{value, scope, n_denominator}} triple"
            )
        # v2 leftovers must be absent.
        assert "hallucination_rate_answerable" not in payload["answer_quality"]
        # Sample-size table is present and complete.
        assert set(payload["sample_sizes"].keys()) == {
            "n_examples",
            "n_answered",
            "n_judged",
            "n_refused",
            "n_answerable_answered",
        }
        # Companion refusal metric.
        assert "out_of_scope_answered" in payload["refusal_calibration"]

    def test_console_uses_the_v3_scoped_hallucination_label(self, tmp_path: Path) -> None:
        """The console must display the v3-scoped name with the denominator,
        never the bare v1 name and never the v2 ``(answerable only)`` label
        (which doesn't say ``n=…``). Load-bearing: no chiffre-without-denominator
        can leak into a demo screenshot."""
        console, _ = _run_stub_and_capture(tmp_path)
        assert "hallucination_rate (answerable_and_answered, n=" in console, (
            "console must use the v3-scoped label including the denominator"
        )
        # The v2 label must not survive.
        assert "hallucination_rate (answerable only)" not in console, (
            "console still shows the v2 label without a denominator — v3 rename incomplete"
        )
        # A bare row without qualifier is also forbidden.
        bare_v1_row = re.search(r"hallucination_rate\s+\|\s*\d", console)
        assert bare_v1_row is None

    def test_console_shows_denominators_on_every_judge_track_metric(self, tmp_path: Path) -> None:
        """faithfulness and citation_accuracy also carry their v3 scope +
        denominator in the console — a reader cannot quote 0.792 without
        seeing the ``(answered, n=12)`` context."""
        console, _ = _run_stub_and_capture(tmp_path)
        assert "faithfulness (answered, n=" in console
        assert "citation_accuracy (answered, n=" in console

    def test_console_shows_out_of_scope_answered_companion(self, tmp_path: Path) -> None:
        """The v2 companion metric survives into v3 unchanged and must appear
        in the console — parity with JSON."""
        console, _ = _run_stub_and_capture(tmp_path)
        assert "out_of_scope_answered" in console

    def test_every_v3_json_metric_appears_in_the_console(self, tmp_path: Path) -> None:
        """The strongest invariant: every user-visible metric name in the v3
        JSON scorecard must resolve to a labelled row in the console output.
        No console-only vocabulary, no JSON-only vocabulary — one source of truth.
        """
        console, _payload = _run_stub_and_capture(tmp_path)
        assert any(k in console for k in ("p50", "p95", "p99")), (
            "console must expose latency percentiles"
        )
        # Every explicit v3 metric name (or the console-side equivalent) must
        # appear as a row somewhere in the printed table.
        must_appear_in_console = {
            "hallucination_rate",
            "out_of_scope_answered",
            "faithfulness",
            "citation_accuracy",
            "refusal_precision",
            "refusal_recall",
        }
        for name in must_appear_in_console:
            assert name in console, (
                f"JSON metric {name!r} missing from console output — "
                f"console/JSON labels have drifted; both must be sourced from one place"
            )


# --------------------------------------------------------------------------------------
# Related sanity: the CLI verdict label for q-015-like cases is the honest one.
# --------------------------------------------------------------------------------------


def test_cli_verdict_labels_do_not_claim_hallucination_without_evidence(
    tmp_path: Path,
) -> None:
    """The confusion-matrix verdict for ``(unrefused, out-of-scope)`` was
    renamed from ``FN (hallucinated on out-of-scope)`` to ``FN (didn't refuse
    out-of-scope)`` because the judge's opinion of the text is separately
    stored under ``per_audit`` — the confusion-matrix label alone doesn't
    justify the word 'hallucinated'."""
    console, _ = _run_stub_and_capture(tmp_path)
    assert "hallucinated on out-of-scope" not in console, (
        "verdict label must not claim hallucination — that's a judge verdict, "
        "not a confusion-matrix cell"
    )
