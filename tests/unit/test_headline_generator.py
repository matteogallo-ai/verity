"""Headline scorecard round-trip: build_headline_dict + write + README render.

The generator script is what the mission requires ("README results table is
generated from that JSON by a script — never hand-typed"). This test asserts:

1. :func:`build_headline_dict` extracts every headline field from a
   ``HarnessResult`` verbatim (no rescaling, no computed composite).
2. The generator's :func:`render_block` emits Markdown containing the exact
   raw numeric values from the JSON — no client-side reformatting that could
   silently drift the published number.
3. Running the generator twice on the same JSON is a no-op (idempotence).
4. Missing markers in the README fail LOUDLY (the whole point).
"""

from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from verity.eval import build_headline_dict, write_headline_scorecard
from verity.eval.harness import HarnessResult, JudgeCallStamp, PerExampleRefusal
from verity.types import (
    Answer,
    AnswerMetrics,
    Confidence,
    LatencyMetrics,
    PerExampleAudit,
    Provider,
    RetrievalMetrics,
    Scorecard,
    UsageStats,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_headline_results.py"


def _load_generator():  # type: ignore[no-untyped-def]
    """Import the CLI script as a module so we can call render_block / render_readme."""
    spec = importlib.util.spec_from_file_location("headline_gen", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture_result() -> HarnessResult:
    """Assemble a HarnessResult that mirrors a realistic v1.0.0 shape."""
    return HarnessResult(
        scorecard=Scorecard(
            git_sha="abc1234",
            created_at=datetime(2026, 9, 7, 12, 0, tzinfo=UTC),
            dataset="questions",
            n_examples=16,
            retrieval=RetrievalMetrics(
                precision_at_k=0.148,
                recall_at_k=1.0,
                ndcg_at_k=0.95,
                k=8,
            ),
            answer=AnswerMetrics(
                faithfulness=0.94,
                citation_accuracy=0.90,
                hallucination_rate=0.06,
                refusal_precision=1.0,
                refusal_recall=0.80,
            ),
            latency=LatencyMetrics(p50_ms=850.0, p95_ms=2400.0, p99_ms=3100.0),
            cost_per_query_usd=0.09,
            cold_start_ms=6200.0,
        ),
        embedding_model="BAAI/bge-small-en-v1.5",
        judge_model="claude-sonnet-4-6",
        agent_model="claude-sonnet-4-6",
        answers=[
            Answer(
                query=f"q{i}",
                text="stub",
                confidence=Confidence(score=0.9, refused=False, rationale=""),
                hits_used=(),
                usage=UsageStats(cost_usd=0.05),
                provider_used=Provider.ANTHROPIC,
                model_used="claude-sonnet-4-6",
            )
            for i in range(16)
        ],
        per_refusal=[
            PerExampleRefusal(
                example_id=f"q-{i:03d}",
                expected_answerable=(i < 11),
                observed_refused=(i >= 11),
            )
            for i in range(16)
        ],
        judge_cost_usd=0.60,
        judge_stamps=[
            JudgeCallStamp(
                example_id=f"q-{i:03d}",
                # Judge is bypassed on refusals (the 5 unanswerable examples)
                # so those calls stamp None — the honest signal "no LLM ran".
                provider=None if i >= 11 else Provider.ANTHROPIC,
                model=None if i >= 11 else "claude-sonnet-4-6",
            )
            for i in range(16)
        ],
        # v3 requires per_audit for the honest per-scope aggregate computation.
        # i in [0..10]: answerable, answered by judge → 11 at-risk. Each carries
        # the per-example metrics that produce v3 aggregates matching the
        # scorecard's field values (all 11 identical for simplicity).
        # i in [11..15]: OOS, refused → judge bypassed, judge_claims=None.
        per_audit=[
            PerExampleAudit(
                example_id=f"q-{i:03d}",
                question=f"q{i}",
                expected_answerable=(i < 11),
                answer_text="stub" if i < 11 else "I don't know.",
                refused=(i >= 11),
                refusal_rationale=("" if i < 11 else "out-of-scope refusal"),
                judge_claims=None if i >= 11 else (),
                answer_metrics=AnswerMetrics(
                    # Judged: match the aggregate values so v3 = the fixture
                    # aggregates. Refusals: judge convention (1.0/1.0/0.0).
                    faithfulness=0.94 if i < 11 else 1.0,
                    citation_accuracy=0.90 if i < 11 else 1.0,
                    hallucination_rate=0.06 if i < 11 else 0.0,
                    refusal_precision=0.0,
                    refusal_recall=0.0,
                ),
                usage=UsageStats(cost_usd=0.05),
            )
            for i in range(16)
        ],
    )


def test_build_headline_dict_carries_full_provenance_and_totals() -> None:
    result = _fixture_result()
    payload = build_headline_dict(result)
    assert payload["schema"] == "verity.scorecard.headline/v3"
    prov = payload["provenance"]
    # Agent summary: all 16 answers stamped claude-sonnet-4-6 → "(16/16)"
    assert prov["agent_model"] == "claude-sonnet-4-6 (16/16)"
    # Judge summary (v3): the 5 refusals bypass the judge, so the count MUST
    # be expressed as "11/16 answered — 5 refusals bypass the judge" so a
    # reader can never confuse the judge call count with the total question count.
    assert prov["judge_model"] == (
        "claude-sonnet-4-6 (11/16 answered — 5 refusals bypass the judge)"
    )
    assert prov["embedding_model"] == "BAAI/bge-small-en-v1.5"
    assert prov["run_sha"] == "abc1234"
    # Total cost = 16 answers * $0.05 + judge $0.60 = $1.40 (exact — never rounded away).
    assert payload["cost"]["total_usd"] == pytest.approx(1.4)
    assert payload["cost"]["agent_usd"] == pytest.approx(0.8)
    assert payload["cost"]["judge_usd"] == pytest.approx(0.6)
    # Dataset counts derive from per_refusal — not from raw n_examples.
    assert payload["dataset"]["n_answerable"] == 11
    assert payload["dataset"]["n_unanswerable"] == 5

    # v3 sample-size table: every denominator is one click away from the scorecard.
    sizes = payload["sample_sizes"]
    assert sizes == {
        "n_examples": 16,
        "n_answered": 11,
        "n_judged": 11,
        "n_refused": 5,
        "n_answerable_answered": 11,
    }

    # v3 answer_quality: each metric is {value, scope, n_denominator}.
    aq = payload["answer_quality"]
    # Faithfulness: scope=answered, n=11 judged answers, all with per-example 0.94.
    assert aq["faithfulness"]["value"] == pytest.approx(0.94)
    assert aq["faithfulness"]["scope"] == "answered"
    assert aq["faithfulness"]["n_denominator"] == 11
    # Citation accuracy: same scope, per-example 0.90.
    assert aq["citation_accuracy"]["value"] == pytest.approx(0.90)
    assert aq["citation_accuracy"]["scope"] == "answered"
    assert aq["citation_accuracy"]["n_denominator"] == 11
    # Hallucination: scope=answerable_and_answered, at-risk set = 11.
    assert aq["hallucination_rate"]["value"] == pytest.approx(0.06)
    assert aq["hallucination_rate"]["scope"] == "answerable_and_answered"
    assert aq["hallucination_rate"]["n_denominator"] == 11

    # Refusal calibration: precision/recall derived from per_refusal.
    # 5 refusals, all on OOS (i>=11) → TP=5, FN=0 → precision=1.0, recall=1.0.
    assert payload["refusal_calibration"]["refusal_precision"] == pytest.approx(1.0)
    assert payload["refusal_calibration"]["refusal_recall"] == pytest.approx(1.0)
    # Companion field (v2): in this fixture per_refusal has i>=11 all refused,
    # so 0 out-of-scope answers were shipped (out of 5).
    ooa = payload["refusal_calibration"]["out_of_scope_answered"]
    assert ooa == {"count": 0, "total": 5}
    # dataset.sha256 is None when no dataset_path is passed — surfaces honestly.
    assert payload["dataset"]["sha256"] is None


def test_agent_and_judge_summaries_reflect_a_mixed_provider_split() -> None:
    """The load-bearing invariant of the micro-patch: when the router fell back
    to a second provider for some requests, the scorecard's ``agent_model`` /
    ``judge_model`` fields honestly split the count — never a boot-time default
    that erases the fallback."""
    base = _fixture_result()
    # Rebuild answers: 15 anthropic + 1 openai fallback.
    mixed_answers = [
        Answer(
            query=f"q{i}",
            text="stub",
            confidence=Confidence(score=0.9, refused=False, rationale=""),
            hits_used=(),
            usage=UsageStats(cost_usd=0.05),
            provider_used=Provider.OPENAI if i == 15 else Provider.ANTHROPIC,
            model_used="gpt-4.1-mini" if i == 15 else "claude-sonnet-4-6",
        )
        for i in range(16)
    ]
    mixed_judge = [
        JudgeCallStamp(
            example_id=f"q-{i:03d}",
            provider=None if i >= 11 else (Provider.OPENAI if i == 10 else Provider.ANTHROPIC),
            model=None if i >= 11 else ("gpt-4.1-mini" if i == 10 else "claude-sonnet-4-6"),
        )
        for i in range(16)
    ]
    result = HarnessResult(
        scorecard=base.scorecard,
        embedding_model=base.embedding_model,
        judge_model=base.judge_model,
        agent_model=base.agent_model,
        answers=mixed_answers,
        per_refusal=list(base.per_refusal),
        judge_cost_usd=base.judge_cost_usd,
        judge_stamps=mixed_judge,
        per_audit=list(base.per_audit),
    )
    payload = build_headline_dict(result)
    assert payload["provenance"]["agent_model"] == "claude-sonnet-4-6 (15/16), gpt-4.1-mini (1/16)"
    # v3 judge summary with refusal-skip disclosure: 11 judge calls out of 16
    # examples, 5 refusals bypass. One of the 11 calls fell back to OpenAI.
    assert payload["provenance"]["judge_model"] == (
        "claude-sonnet-4-6 (10/16 answered — 5 refusals bypass the judge), "
        "gpt-4.1-mini (1/16 answered — 5 refusals bypass the judge)"
    )


def test_dataset_sha256_is_included_when_path_provided(tmp_path: Path) -> None:
    """Dataset identity locked in the JSON so a live run can be verified as
    having scored the same file the stub validated on."""
    import hashlib

    dataset = tmp_path / "questions.jsonl"
    dataset.write_bytes(b'{"id":"q-001","question":"x"}\n')
    expected = hashlib.sha256(dataset.read_bytes()).hexdigest()

    result = _fixture_result()
    payload = build_headline_dict(result, dataset_path=dataset)
    assert payload["dataset"]["sha256"] == expected


def test_render_block_shows_verbatim_headline_numbers() -> None:
    result = _fixture_result()
    payload = build_headline_dict(result)
    generator = _load_generator()
    block = generator.render_block(payload)
    assert "1.000" in block  # refusal_recall (fixture: 5 OOS, all refused)
    assert "0.940" in block  # faithfulness value
    assert "0.950" in block  # ndcg
    assert "claude-sonnet-4-6" in block  # provenance
    assert "abc1234" in block  # commit
    assert "$1.4000" in block  # total cost (4-decimal precision)
    # v3 scope + denominator visible in every judge-track row.
    assert "(answered, n=11)" in block
    assert "(answerable_and_answered, n=11)" in block
    assert "refusals bypass the judge" in block  # judge summary honesty


def test_write_headline_scorecard_round_trips(tmp_path: Path) -> None:
    result = _fixture_result()
    out = write_headline_scorecard(result, tmp_path / "live.json")
    parsed = json.loads(out.read_text(encoding="utf-8"))
    assert parsed["schema"] == "verity.scorecard.headline/v3"
    assert parsed["cost"]["total_usd"] == pytest.approx(1.4)


def test_generator_is_idempotent(tmp_path: Path) -> None:
    """Running the generator twice is a no-op — no silent README drift."""
    result = _fixture_result()
    scorecard_path = tmp_path / "live.json"
    write_headline_scorecard(result, scorecard_path)
    readme = tmp_path / "README.md"
    readme.write_text(
        "# top\n\n## Measured results\n\n<!-- RESULTS:START -->\n<!-- RESULTS:END -->\n\nend\n",
        encoding="utf-8",
    )
    generator = _load_generator()
    block = generator.render_block(json.loads(scorecard_path.read_text()))
    first = generator.render_readme(readme.read_text(encoding="utf-8"), block)
    second = generator.render_readme(first, block)
    assert first != readme.read_text(encoding="utf-8")
    assert second == first


def test_generator_fails_loudly_without_markers(tmp_path: Path) -> None:
    result = _fixture_result()
    scorecard_path = tmp_path / "live.json"
    write_headline_scorecard(result, scorecard_path)
    readme = tmp_path / "README.md"
    readme.write_text("no markers here\n", encoding="utf-8")
    generator = _load_generator()
    with pytest.raises(SystemExit):
        generator.render_readme(
            readme.read_text(encoding="utf-8"),
            generator.render_block(json.loads(scorecard_path.read_text())),
        )


@pytest.mark.parametrize(
    ("cold_start", "must_contain", "must_not_contain"),
    [
        (None, "no warmup", "warm;"),
        (1234.5, "warm; cold start `1234.5 ms` measured separately", "no warmup"),
    ],
)
def test_latency_row_says_whether_cold_start_is_included(
    cold_start: float | None, must_contain: str, must_not_contain: str
) -> None:
    """A run without warmup (cold_start = null) must not be labelled "warm"."""
    generator = _load_generator()
    payload = json.loads(Path("scorecards/live-v1.0.0.json").read_text(encoding="utf-8"))
    payload["latency_ms"]["cold_start"] = cold_start
    latency_row = next(
        line
        for line in generator.render_block(payload).splitlines()
        if line.startswith("| Latency")
    )
    assert must_contain in latency_row
    assert must_not_contain not in latency_row


def test_answer_quality_rows_state_macro_aggregation() -> None:
    generator = _load_generator()
    payload = json.loads(Path("scorecards/live-v1.0.0.json").read_text(encoding="utf-8"))
    block = generator.render_block(payload)
    faith = next(line for line in block.splitlines() if line.startswith("| Faithfulness"))
    cit = next(line for line in block.splitlines() if line.startswith("| Citation accuracy"))
    assert "per-question mean" in faith and "per-question mean" in cit
    assert "claim-level, judge ran" not in faith


def test_committed_readme_matches_committed_scorecard(tmp_path: Path) -> None:
    """The real README's results block is what the committed scorecard renders."""
    generator = _load_generator()
    readme = tmp_path / "README.md"
    readme.write_text((REPO_ROOT / "README.md").read_text(encoding="utf-8"), encoding="utf-8")
    assert generator.main(["--check", "--readme", str(readme)]) == 0


def test_check_mode_never_writes_the_readme(tmp_path: Path) -> None:
    """``--check`` compares and exits 1 on drift; the file on disk is left as is."""
    generator = _load_generator()
    readme = tmp_path / "README.md"
    stale = (REPO_ROOT / "README.md").read_text(encoding="utf-8").replace("| 0.722", "| 0.999")
    readme.write_text(stale, encoding="utf-8")
    assert generator.main(["--check", "--readme", str(readme)]) == 1
    assert readme.read_text(encoding="utf-8") == stale


# --- system card block ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("k", "n", "low", "high"),
    [
        # Reference values of the Wilson (1927) score interval at z = 1.96.
        (5, 10, 0.236593, 0.763407),
        (0, 10, 0.0, 0.277533),
        (10, 10, 0.722467, 1.0),
        (17, 22, 0.565600, 0.898770),
    ],
)
def test_wilson_interval_matches_reference_values(k: int, n: int, low: float, high: float) -> None:
    generator = _load_generator()
    assert generator.wilson_interval(k, n) == pytest.approx((low, high), abs=1e-6)


def test_wilson_interval_is_none_without_observations_and_rejects_impossible_counts() -> None:
    generator = _load_generator()
    assert generator.wilson_interval(0, 0) is None
    for k, n in ((-1, 5), (6, 5)):
        with pytest.raises(ValueError):
            generator.wilson_interval(k, n)


def _committed_scorecard() -> dict:  # type: ignore[type-arg]
    return json.loads((REPO_ROOT / "scorecards" / "live-v1.0.0.json").read_text(encoding="utf-8"))


def _card_rows(payload: dict) -> dict[str, list[str]]:  # type: ignore[type-arg]
    """Table rows of the rendered system card block, keyed by their first cell."""
    generator = _load_generator()
    block = generator.render_system_card_block(payload, scorecard_path=generator.DEFAULT_SCORECARD)
    rows: dict[str, list[str]] = {}
    for line in block.splitlines():
        if line.startswith("| ") and not line.startswith("|---"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            rows[cells[0]] = cells[1:]
    return rows


def _row_starting(rows: dict[str, list[str]], prefix: str) -> list[str]:
    return next(cells for name, cells in rows.items() if name.startswith(prefix))


def test_system_card_recovers_counts_from_the_artefact_and_gives_intervals() -> None:
    rows = _card_rows(_committed_scorecard())
    # 5 out-of-scope questions, 4 refused; all 4 refusals were out of scope.
    assert rows["Refusal recall"] == ["0.800", "4/5", "[0.376, 0.964]"]
    assert rows["Refusal precision"] == ["1.000", "4/4", "[0.510, 1.000]"]
    # A per-answer hallucination flag is 0 or 1, so the rate gives k back.
    assert rows["Hallucination rate ↓ (judge)"] == ["0.000", "0/11", "[0.000, 0.259]"]
    # A mean of per-question fractions has no k: n is shown, no interval is invented.
    assert rows["Citation accuracy, per-question mean (judge)"] == ["0.722", "n=12", "no interval"]


def test_system_card_slos_are_approved_and_say_what_was_not_measured() -> None:
    rows = _card_rows(_committed_scorecard())
    slos = [_row_starting(rows, prefix) for prefix in ("p95", "Error", "Cost")]
    assert all(cells[0].endswith("(approved)") for cells in slos)
    assert slos[1][1:] == ["not recorded by this scorecard", "not measured"]
    assert slos[2][2] == "met"


def test_a_p95_over_target_is_published_as_not_met_with_its_cause() -> None:
    """Recomputed from the run's per-answer latencies, which must reproduce the
    scorecard p95; a miss names the answers over target instead of guessing a cause."""
    rows = _card_rows(_committed_scorecard())
    cells = _row_starting(rows, "p95 latency of one answer, cold start included")
    assert cells[0] == "≤ 15,000 ms (approved)"
    assert cells[1] == "32526.5 ms (n=16, no warmup: the first answer loads the models)"
    assert cells[2].startswith("**SLO not met**: answers over 15,000 ms: q-003 37684.8 ms, ")
    assert "q-001 30807.0 ms (first answer, model loading)" in cells[2]


def test_the_warm_p95_leaves_out_the_first_answer_and_still_misses() -> None:
    rows = _card_rows(_committed_scorecard())
    warm = _row_starting(rows, "p95 latency of one answer, warm")
    assert warm[1] == "22410.1 ms (n=15, first answer left out)"
    assert warm[2].startswith("**SLO not met**: answers over 15,000 ms: q-003 37684.8 ms, ")
    assert "q-001" not in warm[2]
    # The run kept no per-stage timings or retry counts: the cause of the slow
    # answers other than the first is said to be unknown, not guessed.
    assert "not attributed" in warm[2]


def test_without_a_run_artefact_the_warm_p95_comes_from_a_warmed_scorecard() -> None:
    payload = _committed_scorecard()
    payload["provenance"]["run_sha"] = "0000000"  # no such artefact
    payload["latency_ms"] = {"cold_start": 8460.1, "p50": 5445.7, "p95": 8949.3, "p99": 10929.3}
    rows = _card_rows(payload)
    assert _row_starting(rows, "p95 latency of one answer, warm")[1:] == [
        "8949.3 ms (n=16, after one warmup answer)",
        "met",
    ]
    assert _row_starting(rows, "p95 latency of one answer, cold start included")[1:] == [
        "not measured as one sample: the first answer after boot, timed apart, took 8460.1 ms",
        "not measured",
    ]


def test_a_run_artefact_that_does_not_reproduce_the_scorecard_p95_is_refused() -> None:
    payload = _committed_scorecard()
    payload["latency_ms"]["p95"] = 1000.0
    with pytest.raises(SystemExit, match="does not reproduce"):
        _card_rows(payload)


def test_system_card_reads_counts_stored_in_a_newer_scorecard() -> None:
    """A scorecard that stores k/n per metric and an errored count is rendered from
    those counts, so the card follows whichever scorecard the README renders."""
    payload = _committed_scorecard()
    payload["refusal"] = payload.pop("refusal_calibration")
    payload["refusal"]["refusal_precision"] = {"value": 0.772727, "k": 17, "n": 22}
    payload["sample_sizes"]["n_errored"] = 1
    rows = _card_rows(payload)
    assert rows["Refusal precision"] == ["0.773", "17/22", "[0.566, 0.899]"]
    assert _row_starting(rows, "Error")[1:] == [
        "0.062 (1/16, 95 % CI [0.011, 0.283])",
        "**SLO not met**: 1 of 16 requests errored; the scorecard does not record why",
    ]


def test_per_claim_counts_in_a_newer_scorecard_get_a_per_claim_row() -> None:
    payload = _committed_scorecard()
    payload["answer_quality"]["faithfulness"] = {
        "scope": "answered",
        "micro": {"value": 0.990826, "k": 108, "n_claims": 109},
    }
    rows = _card_rows(payload)
    assert rows["Faithfulness per claim (judge)"] == ["0.991", "108/109", "[0.950, 0.998]"]


def test_a_missing_card_marker_writes_neither_file(tmp_path: Path) -> None:
    generator = _load_generator()
    readme = tmp_path / "README.md"
    stale = (REPO_ROOT / "README.md").read_text(encoding="utf-8").replace("| 0.722", "| 0.999")
    readme.write_text(stale, encoding="utf-8")
    card = tmp_path / "SYSTEM_CARD.md"
    card.write_text("no markers\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        generator.main(["--readme", str(readme), "--system-card", str(card)])
    assert readme.read_text(encoding="utf-8") == stale
    with pytest.raises(SystemExit, match="not found"):
        generator.main(["--readme", str(readme), "--system-card", str(tmp_path / "absent.md")])


def test_a_refusal_value_that_disagrees_with_its_records_fails_loudly() -> None:
    payload = _committed_scorecard()
    payload["refusal_calibration"]["refusal_recall"] = 0.9
    with pytest.raises(SystemExit):
        _card_rows(payload)


def test_check_covers_the_system_card_and_never_writes_it(tmp_path: Path) -> None:
    generator = _load_generator()
    readme = tmp_path / "README.md"
    readme.write_text((REPO_ROOT / "README.md").read_text(encoding="utf-8"), encoding="utf-8")
    card = tmp_path / "SYSTEM_CARD.md"
    committed = (REPO_ROOT / "docs" / "SYSTEM_CARD.md").read_text(encoding="utf-8")
    args = ["--readme", str(readme), "--system-card", str(card)]
    card.write_text(committed, encoding="utf-8")
    # The source link is relative to the card, so a copy elsewhere is regenerated first.
    assert generator.main(args) == 0
    args.insert(0, "--check")
    assert generator.main(args) == 0
    regenerated = card.read_text(encoding="utf-8")
    stale = regenerated.replace("| 4/5 |", "| 5/5 |")
    assert stale != regenerated
    card.write_text(stale, encoding="utf-8")
    assert generator.main(args) == 1
    assert card.read_text(encoding="utf-8") == stale
