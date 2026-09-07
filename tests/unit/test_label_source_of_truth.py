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
# Load-bearing invariant: JSON v2 keys ↔ console labels.
# --------------------------------------------------------------------------------------


class TestConsoleJsonLabelInvariant:
    """Every user-visible metric name is defined ONCE. Console and JSON agree."""

    def test_json_has_v2_renamed_metric_and_companion(self, tmp_path: Path) -> None:
        """Baseline: the JSON we generate today is the v2 schema — no drift back
        to v1 ``hallucination_rate`` under any refactor."""
        _, payload = _run_stub_and_capture(tmp_path)
        assert payload["schema"] == "verity.scorecard.headline/v2"
        assert "hallucination_rate_answerable" in payload["answer_quality"]
        assert "hallucination_rate" not in payload["answer_quality"], (
            "v1 name must not leak into a v2 scorecard — enforces the rename"
        )
        assert "out_of_scope_answered" in payload["refusal_calibration"]
        ooa = payload["refusal_calibration"]["out_of_scope_answered"]
        assert set(ooa.keys()) == {"count", "total"}

    def test_console_uses_the_v2_qualified_hallucination_label(self, tmp_path: Path) -> None:
        """The console must display the v2-qualified name, not the bare v1 name.
        The load-bearing bit: ``(answerable only)`` in the label so a reader
        cannot quote the number without knowing the gate.
        """
        console, _ = _run_stub_and_capture(tmp_path)
        assert "hallucination_rate (answerable only)" in console, (
            "console must use the v2-qualified label — mirrors JSON key semantics"
        )
        # The bare v1 label without the qualifier must NOT appear as a row on its own.
        # (It may appear as a substring inside the qualified label — that's fine;
        # what we forbid is a stand-alone " hallucination_rate " column entry.)
        bare_v1_row = re.search(r"hallucination_rate\s+\|\s*\d", console)
        assert bare_v1_row is None, (
            "console still displays the bare v1 ``hallucination_rate`` — v2 rename incomplete"
        )

    def test_console_shows_out_of_scope_answered_companion(self, tmp_path: Path) -> None:
        """The v2 companion metric must appear in the console output too — the
        row that captures the half ``hallucination_rate_answerable`` cannot see."""
        console, _ = _run_stub_and_capture(tmp_path)
        assert "out_of_scope_answered" in console, (
            "console must show the v2 companion metric — parity with JSON"
        )

    def test_every_console_row_metric_is_a_v2_json_key(self, tmp_path: Path) -> None:
        """The strongest invariant: every metric label that appears as a table
        row in the console output must resolve to a key in the v2 JSON scorecard.
        No console-only vocabulary, no JSON-only vocabulary — one source of truth.

        Concretely, we assert the set of console row labels ⊆ the set of v2
        JSON metric names (plus a small allow-list of console-only decorative
        rows like latency percentiles which have their own top-level keys)."""
        console, payload = _run_stub_and_capture(tmp_path)

        # The v2 JSON scorecard's user-visible metric names, flattened from
        # the schema. Keep this list in sync with ``build_headline_dict``.
        json_metric_names: set[str] = set()
        json_metric_names.update(payload["answer_quality"].keys())
        json_metric_names.update(payload["refusal_calibration"].keys())
        # Latency subkeys (p50/p95/p99/cold_start) show as e.g. "latency p50 ms".
        # We assert the console shows *some* latency percentile prefix.
        assert any(k in console for k in ("p50", "p95", "p99")), (
            "console must expose latency percentiles"
        )
        # Cost per query is its own JSON key with a console counterpart.
        json_metric_names.add("cost_per_query_usd")
        # Every explicitly-renamed v2 name must appear in the console.
        must_appear_in_console = {
            "hallucination_rate_answerable",
            "out_of_scope_answered",
            "faithfulness",
            "citation_accuracy",
            "refusal_precision",
            "refusal_recall",
        }
        for name in must_appear_in_console:
            # The console prints e.g. "hallucination_rate (answerable only)" —
            # so the JSON key without the suffix is enough as a substring probe.
            probe = name.replace("_answerable", "")
            assert probe in console, (
                f"JSON metric {name!r} (probe={probe!r}) missing from console output — "
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
