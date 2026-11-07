"""Console-vs-JSON label consistency contract.

A pre-release audit exposed this gap: the JSON scorecard was renamed to
``hallucination_rate_answerable`` + ``out_of_scope_answered`` (schema v2),
but the Rich console table in :func:`verity.cli._print_scorecard` still
printed the pre-v2 name ``hallucination_rate`` with no companion row. A
reader looking at the console got a different vocabulary than a reader
looking at the JSON — the same run, two different label sets.

These tests LOCK the invariant: every headline label that appears in the
CLI console output must exist as a key in the JSON scorecard schema, and
every user-visible metric in the JSON must also appear in the console table.
There is one source of truth for what a metric is called.

Zero LLM cost — stub agent + stub judge on the real 16-question dataset.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from verity.cli import app
from verity.config import get_settings

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _clean(text: str) -> str:
    """Strip ANSI + collapse whitespace so substring checks are terminal-agnostic."""
    return re.sub(r"\s+", " ", _ANSI_RE.sub("", text)).strip()


@pytest.fixture(scope="module")
def stub_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, dict]:
    """Run ``verity eval run --ci`` with a stub scorecard once for the module, return
    (console, JSON). Every test here reads the same run: one stub eval over the full
    dataset takes minutes on a CPU runner, and each test repeating it made the slow
    lane outlast the CI job."""
    tmp_path = tmp_path_factory.mktemp("stub_run")
    scorecard = tmp_path / "sc.json"
    runs_dir = tmp_path / "runs"
    # The conftest's function-scoped memory-store fixture does not reach a module
    # fixture; without the variable the run would open the default pgvector store.
    get_settings.cache_clear()
    result = runner.invoke(
        app,
        ["eval", "run", "--ci", "--runs-dir", str(runs_dir), "--scorecard-json", str(scorecard)],
        env={"COLUMNS": "200", "TERM": "dumb", "NO_COLOR": "1", "VERITY_STORE": "memory"},
    )
    get_settings.cache_clear()
    assert result.exit_code == 0, f"stub run failed: {result.output}"
    assert scorecard.is_file()
    payload = json.loads(scorecard.read_text(encoding="utf-8"))
    return _clean(result.output), payload


# --------------------------------------------------------------------------------------
# Load-bearing invariant: JSON v3 keys ↔ console labels.
# --------------------------------------------------------------------------------------


@pytest.mark.slow
class TestConsoleJsonLabelInvariant:
    """Every user-visible metric name + denominator is defined ONCE.
    Console and JSON agree on vocabulary AND on scope/n_denominator."""

    def test_json_is_v5_with_both_aggregations(self, stub_run: tuple[str, dict]) -> None:
        """Faithfulness and citation accuracy carry a macro and a micro figure, each
        with its denominator; each proportion also carries its k and Wilson interval."""
        _, payload = stub_run
        assert payload["schema"] == "verity.scorecard.headline/v6"
        for name in ("faithfulness", "citation_accuracy"):
            entry = payload["answer_quality"][name]
            assert set(entry["macro"]) == {"value", "n_questions"}
            assert set(entry["micro"]) == {"value", "k", "n_claims", "ci95_wilson"}
        assert set(payload["answer_quality"]["hallucination_rate"]) == {
            "value",
            "k",
            "scope",
            "n_questions",
            "ci95_wilson",
        }
        assert set(payload["sample_sizes"]) == {
            "n_examples",
            "n_errored",
            "n_not_run",
            "n_answered",
            "n_judged",
            "n_judged_without_claims",
            "n_claims",
            "n_refused",
            "n_answerable_answered",
        }
        assert "out_of_scope_answered" in payload["refusal"]

    def test_console_uses_the_v3_scoped_hallucination_label(
        self, stub_run: tuple[str, dict]
    ) -> None:
        """The console must display the v3-scoped name with the denominator,
        never the bare v1 name and never the v2 ``(answerable only)`` label
        (which doesn't say ``n=…``). Load-bearing: no number-without-denominator
        can leak into a demo screenshot."""
        console, _ = stub_run
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

    def test_console_shows_denominators_on_every_judge_track_metric(
        self, stub_run: tuple[str, dict]
    ) -> None:
        """faithfulness and citation_accuracy also carry their v3 scope +
        denominator in the console — a reader cannot quote 0.792 without
        seeing the ``(answered, n=12)`` context."""
        console, _ = stub_run
        assert "faithfulness (answered, n=" in console
        assert "citation_accuracy (answered, n=" in console
        assert "faithfulness micro (claims, n=" in console
        assert "citation_accuracy micro (claims, n=" in console

    def test_console_shows_out_of_scope_answered_companion(
        self, stub_run: tuple[str, dict]
    ) -> None:
        """The v2 companion metric survives into v3 unchanged and must appear
        in the console — parity with JSON."""
        console, _ = stub_run
        assert "out_of_scope_answered" in console

    def test_every_v3_json_metric_appears_in_the_console(self, stub_run: tuple[str, dict]) -> None:
        """The strongest invariant: every user-visible metric name in the v3
        JSON scorecard must resolve to a labelled row in the console output.
        No console-only vocabulary, no JSON-only vocabulary — one source of truth.
        """
        console, _payload = stub_run
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


@pytest.mark.slow
def test_cli_verdict_labels_do_not_claim_hallucination_without_evidence(
    stub_run: tuple[str, dict],
) -> None:
    """The confusion-matrix verdict for ``(unrefused, out-of-scope)`` was
    renamed from ``FN (hallucinated on out-of-scope)`` to ``FN (didn't refuse
    out-of-scope)`` because the judge's opinion of the text is separately
    stored under ``per_audit`` — the confusion-matrix label alone doesn't
    justify the word 'hallucinated'."""
    console, _ = stub_run
    assert "hallucinated on out-of-scope" not in console, (
        "verdict label must not claim hallucination — that's a judge verdict, "
        "not a confusion-matrix cell"
    )


@pytest.mark.slow
def test_the_cli_scorecard_is_the_one_its_artefact_rebuilds(stub_run: tuple[str, dict]) -> None:
    """``eval run --scorecard-json`` and ``generate_headline_results.py`` produce the
    same file: the CLI scorecard names its artefact and the threshold the run used."""
    _, payload = stub_run
    assert payload["source_artefact"] is not None
    assert payload["refusal"]["confidence_threshold"] is not None

    from verity.eval.headline import build_headline_dict_from_eval_run
    from verity.eval.run_store import load_run

    rebuilt = build_headline_dict_from_eval_run(
        load_run(Path(payload["source_artefact"])), source_artefact=payload["source_artefact"]
    )
    assert rebuilt == payload
