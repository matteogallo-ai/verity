"""CLI surface is stable and every command runs (scaffold bodies exit 0)."""

from __future__ import annotations

import json
import os
import re
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from verity import __version__
from verity.cli import MISSING_KEY_MESSAGE, app
from verity.config import Settings, get_settings
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


@pytest.fixture(scope="module")
def _stub_harness_results() -> dict[str, object]:
    return {}


@pytest.fixture
def shared_stub_harness(
    _stub_harness_results: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Serve one stub harness run to every test of the module that asks for it.

    These tests pin what the CLI does with a finished run (floors, exit codes, what
    it persists); the run itself is the same stub evaluation of the full dataset,
    minutes on a CI runner, so it is computed once. The cached run ignores the
    arguments of later calls (budget cap, warmup): a test that pins those, or that
    changes the run (an outage inside it), does not request this fixture."""
    from verity.eval import AgentEvaluator

    original = AgentEvaluator.run

    async def _once(self: AgentEvaluator, *args: object, **kwargs: object) -> object:
        if "run" not in _stub_harness_results:
            _stub_harness_results["run"] = await original(self, *args, **kwargs)  # type: ignore[arg-type]
        return _stub_harness_results["run"]

    monkeypatch.setattr(AgentEvaluator, "run", _once)


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


@pytest.mark.slow
def test_eval_run_ci_exits_clean(tmp_path: Path, shared_stub_harness: None) -> None:
    """Hermetic on ``--runs-dir tmp_path`` so the test never writes to the real
    ``datasets/eval/runs/`` tree. A pre-release audit exposed that the two
    prior tests running without ``--runs-dir`` would clobber whatever sat at
    the default path — including a committed live-run audit artefact.
    """
    result = runner.invoke(app, ["eval", "run", "--ci", "--runs-dir", str(tmp_path)])
    assert result.exit_code == 0


@pytest.mark.slow
def test_eval_run_report_only_floors_writes_json_and_exits_zero(
    tmp_path: Path, shared_stub_harness: None
) -> None:
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


@pytest.mark.slow
def test_eval_run_fatal_floors_exit_two_on_stub(tmp_path: Path, shared_stub_harness: None) -> None:
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


def test_ingest_stops_with_exit_2_on_a_schema_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from verity.ingestion.pipeline import IngestionPipeline
    from verity.retrieval.pgvector_store import SchemaNotMigratedError

    async def _raise(self, uris):  # type: ignore[no-untyped-def]
        raise SchemaNotMigratedError("chunks table missing; run `verity db upgrade`")

    monkeypatch.setattr(IngestionPipeline, "ingest", _raise)
    (tmp_path / "a.md").write_text("# A\n\nbody\n")
    result = runner.invoke(app, ["ingest", str(tmp_path)], env={"COLUMNS": "200"})
    assert result.exit_code == 2
    assert "Ingestion stopped" in result.output
    assert "verity db upgrade" in result.output


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
    # The developer's environment must not decide the outcome: no .env file is read
    # and no provider key or VERITY_ variable leaks in from the shell.
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    for name in list(os.environ):
        if name.startswith("VERITY_") or name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("VERITY_STORE", "memory")
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


def _capture_serve(monkeypatch: pytest.MonkeyPatch, args: list[str]) -> tuple[dict, str]:  # type: ignore[type-arg]
    import uvicorn

    import verity.api

    seen: dict = {}  # type: ignore[type-arg]
    monkeypatch.setattr(verity.api, "create_app", lambda **_kw: object())
    monkeypatch.setattr(uvicorn, "run", lambda _app, **kw: seen.update(kw))
    result = runner.invoke(app, ["serve", *args], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    return seen, result.output


def test_serve_binds_loopback_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    seen, output = _capture_serve(monkeypatch, [])
    assert seen["host"] == "127.0.0.1"
    assert "reachable from other machines" not in output


def test_serve_warns_when_bound_to_all_interfaces(monkeypatch: pytest.MonkeyPatch) -> None:
    seen, output = _capture_serve(monkeypatch, ["--host", "0.0.0.0"])
    assert seen["host"] == "0.0.0.0"
    assert "reachable from other machines" in output


def test_serve_refuses_to_start_without_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    from verity.config import get_settings

    monkeypatch.delenv("VERITY_API_KEYS", raising=False)
    monkeypatch.delenv("VERITY_API_AUTH_DISABLED", raising=False)
    get_settings.cache_clear()
    monkeypatch.setattr(uvicorn, "run", lambda *_a, **_kw: pytest.fail("must not serve"))
    try:
        result = runner.invoke(app, ["serve"], env={"COLUMNS": "200"})
    finally:
        get_settings.cache_clear()
    assert result.exit_code == 2
    assert "VERITY_API_KEYS" in result.output


def test_store_option_rejects_an_unknown_backend() -> None:
    result = runner.invoke(app, ["retrieve", "x", "--store", "sqlite"])
    assert result.exit_code != 0
    assert "pgvector" in result.output


def test_store_defaults_to_the_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    from verity.cli import _resolve_store
    from verity.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("VERITY_STORE", "memory")
    try:
        assert _resolve_store(None) == "memory"
        assert _resolve_store("pgvector") == "pgvector"
    finally:
        get_settings.cache_clear()


@pytest.mark.slow
def test_eval_run_with_errored_examples_exits_six(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An outage on one question makes the run incomplete: its metrics cover fewer
    questions than the scorecard's n, so it must not exit as a clean run."""
    from verity.agent.agent import RagAgent
    from verity.types import Answer, AnswerError, ErrorKind

    original = RagAgent.answer
    seen: list[str] = []

    async def _first_fails(self: RagAgent, question: str) -> Answer:
        seen.append(question)
        if len(seen) == 2:  # the first call after the warm-up
            return Answer(
                query=question,
                text="",
                error=AnswerError(kind=ErrorKind.PROVIDER_UNAVAILABLE, detail="down"),
            )
        return await original(self, question)

    monkeypatch.setattr(RagAgent, "answer", _first_fails)
    headline = tmp_path / "headline.json"
    result = runner.invoke(
        app,
        ["eval", "run", "--ci", "--runs-dir", str(tmp_path), "--scorecard-json", str(headline)],
    )
    assert result.exit_code == 6, result.output
    assert "errored" in result.output
    # Kept for its paid examples, under a name that never stands as the run of record.
    assert [p.name for p in tmp_path.glob("eval_*.json") if ".incomplete-" not in p.name] == []
    assert len(list(tmp_path.glob("eval_*.incomplete-*.json"))) == 1
    assert not headline.exists()


@pytest.mark.slow  # the CLI builds the real embedder and reranker
def test_a_budget_abort_persists_what_was_paid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shared_stub_harness: None
) -> None:
    from verity.eval import AgentEvaluator, BudgetExceededError
    from verity.types import IncompleteReason

    original = AgentEvaluator.run

    async def _aborts(self: AgentEvaluator, *args: object, **kwargs: object) -> object:
        done = await original(self, *args, **kwargs)  # type: ignore[arg-type]
        partial = replace(
            done,
            scorecard=done.scorecard.model_copy(
                update={"incomplete_reason": IncompleteReason.BUDGET_EXCEEDED}
            ),
        )
        raise BudgetExceededError("Aborted after q-x", result=partial)

    monkeypatch.setattr(AgentEvaluator, "run", _aborts)
    result = runner.invoke(app, ["eval", "run", "--ci", "--no-warmup", "--runs-dir", str(tmp_path)])
    assert result.exit_code == 4, result.output
    saved = list(tmp_path.glob("eval_*.incomplete-*.json"))
    assert len(saved) == 1, result.output
    run = json.loads(saved[0].read_text(encoding="utf-8"))
    assert run["scorecard"]["incomplete_reason"] == IncompleteReason.BUDGET_EXCEEDED.value
    assert run["per_audit"]


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ([], {"budget_usd_cap": None, "warmup": True}),
        (["--budget-usd", "1.3", "--no-warmup"], {"budget_usd_cap": 1.3, "warmup": False}),
    ],
)
def test_eval_run_hands_its_budget_cap_and_warmup_to_the_evaluator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flags: list[str], expected: dict[str, object]
) -> None:
    """The budget cap is the guard on a paid run and ``--no-warmup`` changes what the
    latency percentiles cover: both must reach the evaluator. The shared stub run
    above ignores them, so this test pins them on their own; ``run`` stops at once,
    before any example."""
    from tests.fakes import without_local_models
    from verity.eval import AgentEvaluator, BudgetExceededError

    without_local_models(monkeypatch)
    seen: dict[str, object] = {}
    original_init = AgentEvaluator.__init__

    def _init(self: AgentEvaluator, *args: object, **kwargs: object) -> None:
        seen["warmup"] = kwargs.get("warmup", "not passed")
        original_init(self, *args, **kwargs)  # type: ignore[arg-type]

    async def _run(self: AgentEvaluator, *args: object, **kwargs: object) -> object:
        seen["budget_usd_cap"] = kwargs.get("budget_usd_cap")
        raise BudgetExceededError("stopped before the first example")

    monkeypatch.setattr(AgentEvaluator, "__init__", _init)
    monkeypatch.setattr(AgentEvaluator, "run", _run)
    result = runner.invoke(app, ["eval", "run", "--ci", "--runs-dir", str(tmp_path), *flags])
    assert result.exit_code == 4, result.output
    assert seen == expected


@pytest.mark.slow
@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ([], {"dataset_name": "questions", "checks": True}),
        (["--naive-baseline"], {"dataset_name": "questions-naive-baseline", "checks": False}),
    ],
)
def test_naive_baseline_runs_an_unchecked_agent_under_its_own_artefact_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flags: list[str], expected: dict[str, object]
) -> None:
    """Both arms run at one commit: the name keeps the second from overwriting the first."""
    from tests.fakes import without_local_models
    from verity.eval import AgentEvaluator, BudgetExceededError

    without_local_models(monkeypatch)
    seen: dict[str, object] = {}
    original_init = AgentEvaluator.__init__

    def _init(self: AgentEvaluator, *args: object, **kwargs: object) -> None:
        seen["dataset_name"] = kwargs.get("dataset_name")
        seen["checks"] = kwargs["agent"]._checks  # type: ignore[attr-defined]
        original_init(self, *args, **kwargs)  # type: ignore[arg-type]

    async def _run(self: AgentEvaluator, *args: object, **kwargs: object) -> object:
        raise BudgetExceededError("stopped before the first example")

    monkeypatch.setattr(AgentEvaluator, "__init__", _init)
    monkeypatch.setattr(AgentEvaluator, "run", _run)
    result = runner.invoke(app, ["eval", "run", "--ci", "--runs-dir", str(tmp_path), *flags])
    assert result.exit_code == 4, result.output
    assert seen == expected


def test_eval_run_refuses_a_sampling_temperature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An eval run records no temperature (ADR 0007); a run at another temperature
    would be compared with runs of record as if only sampling noise separated them."""
    from verity.config import get_settings

    monkeypatch.setenv("VERITY_LLM_TEMPERATURE", "0")
    get_settings.cache_clear()
    result = runner.invoke(app, ["eval", "run", "--ci", "--runs-dir", str(tmp_path)])
    assert result.exit_code == 2
    assert "VERITY_LLM_TEMPERATURE" in result.output
    assert not list(tmp_path.iterdir())


def test_ingest_chunks_with_the_configured_chunker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """VERITY_CHUNKING applies to `verity ingest` as to the API and the eval."""
    from verity.config import get_settings
    from verity.ingestion.article_chunker import ArticleChunker
    from verity.ingestion.pipeline import IngestionPipeline
    from verity.retrieval.pgvector_store import SchemaNotMigratedError

    used: list[object] = []

    async def _record(self, uris):  # type: ignore[no-untyped-def]
        used.append(self._chunker)
        raise SchemaNotMigratedError("stop before any write")

    monkeypatch.setattr(IngestionPipeline, "ingest", _record)
    monkeypatch.setenv("VERITY_CHUNKING", "articles")
    get_settings.cache_clear()
    (tmp_path / "a.md").write_text("# A\n\nbody\n")
    runner.invoke(app, ["ingest", str(tmp_path)], env={"COLUMNS": "200"})
    get_settings.cache_clear()
    assert [type(c) for c in used] == [ArticleChunker]
