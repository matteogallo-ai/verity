"""Headline scorecard round-trip: artefact -> headline dict -> docs/results.md and README.

The generator script is what the mission requires ("README results table is
generated from that JSON by a script — never hand-typed"). This test asserts:

1. :func:`build_headline_dict` extracts every headline field from a
   ``HarnessResult`` verbatim (no rescaling, no computed composite).
2. The generator's :func:`render_block` emits Markdown containing the exact
   raw numeric values from the JSON — no client-side reformatting that could
   silently drift the published number.
3. Running the generator twice on the same JSON is a no-op (idempotence).
4. Missing markers in docs/results.md or the README fail LOUDLY (the whole point).
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
    ClaimVerdict,
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
    """Import the CLI script as a module so we can call render_block / render_results."""
    spec = importlib.util.spec_from_file_location("headline_gen", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_CLAIMS = (
    (ClaimVerdict(claim="a", supported=True, citation_ok=True),) * 45
    + (ClaimVerdict(claim="b", supported=True, citation_ok=False),) * 2
    + (ClaimVerdict(claim="c", supported=False, citation_ok=False),) * 3
)


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
                # 47 of 50 claims supported, 45 of 50 with a supporting citation:
                # micro equals macro because every answer has the same claims.
                judge_claims=None if i >= 11 else _CLAIMS,
                answer_metrics=AnswerMetrics(
                    # Judged: match the aggregate values so v3 = the fixture
                    # aggregates. Refusals: judge convention (1.0/1.0/0.0).
                    faithfulness=0.94 if i < 11 else 1.0,
                    citation_accuracy=0.90 if i < 11 else 1.0,
                    # 0 or 1 per answer, as both judges give it: one of 11 answers.
                    hallucination_rate=1.0 if i == 0 else 0.0,
                    refusal_precision=0.0,
                    refusal_recall=0.0,
                ),
                usage=UsageStats(cost_usd=0.05, input_tokens=100, output_tokens=10),
                judge_usage=UsageStats(cost_usd=0.0 if i >= 11 else 0.6 / 11, input_tokens=7),
                agent_model_used="claude-sonnet-4-6",
                judge_model_used=None if i >= 11 else "claude-sonnet-4-6",
            )
            for i in range(16)
        ],
    )


def test_build_headline_dict_carries_full_provenance_and_totals() -> None:
    result = _fixture_result()
    payload = build_headline_dict(result)
    assert payload["schema"] == "verity.scorecard.headline/v6"
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
        "n_errored": 0,
        "n_not_run": 0,
        "n_answered": 11,
        "n_judged": 11,
        "n_judged_without_claims": 0,
        "n_claims": 550,
        "n_refused": 5,
        "n_answerable_answered": 11,
    }
    assert payload["run"]["agent_tokens"] == {"input": 1600, "output": 160}
    assert payload["run"]["judge_tokens"] == {"input": 112, "output": 0}

    aq = payload["answer_quality"]
    assert aq["faithfulness"]["macro"] == {"value": pytest.approx(0.94), "n_questions": 11}
    assert aq["faithfulness"]["micro"]["value"] == pytest.approx(0.94)
    assert aq["faithfulness"]["micro"]["n_claims"] == 550
    assert aq["citation_accuracy"]["macro"]["value"] == pytest.approx(0.90)
    assert aq["citation_accuracy"]["micro"]["value"] == pytest.approx(0.90)
    assert aq["hallucination_rate"]["value"] == pytest.approx(1 / 11, abs=1e-6)
    assert aq["hallucination_rate"]["n_questions"] == 11

    # Refusal calibration: precision/recall derived from per_refusal.
    # 5 refusals, all on OOS (i>=11) → TP=5, FN=0 → precision=1.0, recall=1.0.
    assert payload["refusal"]["refusal_precision"]["value"] == pytest.approx(1.0)
    assert payload["refusal"]["refusal_recall"]["value"] == pytest.approx(1.0)
    # Companion field (v2): in this fixture per_refusal has i>=11 all refused,
    # so 0 out-of-scope answers were shipped (out of 5).
    ooa = payload["refusal"]["out_of_scope_answered"]
    assert ooa == {
        "count": 0,
        "total": 5,
        "ci95_wilson": [0.0, pytest.approx(0.434482)],
    }
    # dataset.sha256 is None when no dataset_path is passed — surfaces honestly.
    assert payload["dataset"]["sha256"] is None


def test_agent_and_judge_summaries_reflect_a_mixed_provider_split() -> None:
    """The load-bearing invariant of the micro-patch: when the router fell back
    to a second provider for some requests, the scorecard's ``agent_model`` /
    ``judge_model`` fields honestly split the count — never a boot-time default
    that erases the fallback."""
    base = _fixture_result()
    # The agent fell back to OpenAI on q-015, the judge on q-010.
    audits = [
        a.model_copy(
            update={
                "agent_model_used": "gpt-4.1-mini" if i == 15 else a.agent_model_used,
                "judge_model_used": "gpt-4.1-mini" if i == 10 else a.judge_model_used,
            }
        )
        for i, a in enumerate(base.per_audit)
    ]
    result = HarnessResult(**{**base.__dict__, "per_audit": audits})  # type: ignore[arg-type]
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
    # Every judge-track row names its denominator.
    assert "mean over 11 judged answers" in block
    assert "pooled over claims: 517/550" in block
    assert (
        "answered answerable questions (an answer to an unanswerable question is outside it): 1/11"
        in block
    )
    assert "refusals bypass the judge" in block  # judge summary honesty


def test_write_headline_scorecard_round_trips(tmp_path: Path) -> None:
    result = _fixture_result()
    out = write_headline_scorecard(result.as_eval_run(), tmp_path / "live.json")
    parsed = json.loads(out.read_text(encoding="utf-8"))
    assert parsed["schema"] == "verity.scorecard.headline/v6"
    assert parsed["cost"]["total_usd"] == pytest.approx(1.4)


def test_generator_is_idempotent(tmp_path: Path) -> None:
    """Running the generator twice is a no-op — no silent README drift."""
    result = _fixture_result()
    scorecard_path = tmp_path / "live.json"
    write_headline_scorecard(result.as_eval_run(), scorecard_path)
    readme = tmp_path / "README.md"
    readme.write_text(
        "# top\n\n## Measured results\n\n<!-- RESULTS:START -->\n<!-- RESULTS:END -->\n\nend\n",
        encoding="utf-8",
    )
    generator = _load_generator()
    block = generator.render_block(json.loads(scorecard_path.read_text()))
    first = generator.render_results(readme.read_text(encoding="utf-8"), block)
    second = generator.render_results(first, block)
    assert first != readme.read_text(encoding="utf-8")
    assert second == first


def test_generator_fails_loudly_without_markers(tmp_path: Path) -> None:
    result = _fixture_result()
    scorecard_path = tmp_path / "live.json"
    write_headline_scorecard(result.as_eval_run(), scorecard_path)
    readme = tmp_path / "README.md"
    readme.write_text("no markers here\n", encoding="utf-8")
    generator = _load_generator()
    with pytest.raises(SystemExit):
        generator.render_results(
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


def test_answer_quality_rows_publish_both_aggregations() -> None:
    generator = _load_generator()
    payload = json.loads(Path("scorecards/live-v1.0.0.json").read_text(encoding="utf-8"))
    rows = [line for line in generator.render_block(payload).splitlines() if line.startswith("|")]
    for label in ("Faithfulness", "Citation accuracy"):
        assert any(r.startswith(f"| {label}, per question |") for r in rows)
        assert any(r.startswith(f"| {label}, per claim |") for r in rows)


def test_committed_docs_match_committed_scorecard() -> None:
    """docs/results.md and the README's key result are what the committed scorecard renders."""
    assert _load_generator().main(["--check"]) == 0


@pytest.mark.parametrize("marker", ["<!-- RESULTS:END -->", "<!-- KEY-RESULT:END -->"])
def test_check_mode_never_writes_the_docs(tmp_path: Path, marker: str) -> None:
    """``--check`` compares and exits 1 on drift; the files on disk are left as they are."""
    generator = _load_generator()
    readme, results = tmp_path / "README.md", tmp_path / "docs" / "results.md"
    card, scorecard = tmp_path / "SYSTEM_CARD.md", tmp_path / "live.json"
    results.parent.mkdir()
    for copy, name in (
        (readme, "README.md"),
        (results, "docs/results.md"),
        (card, "docs/SYSTEM_CARD.md"),
        (scorecard, "scorecards/live-v1.1.0.json"),
    ):
        copy.write_text((REPO_ROOT / name).read_text(encoding="utf-8"), encoding="utf-8")
    args = ["--readme", str(readme), "--results-doc", str(results)]
    args += ["--system-card", str(card), "--scorecard", str(scorecard)]
    # Links and paths name tmp_path: render once so that only the drift below differs.
    assert generator.main(args) == 0
    assert generator.main(["--check", *args]) == 0
    target = results if marker.startswith("<!-- RESULTS") else readme
    current = target.read_text(encoding="utf-8")
    # A line inside the generated block: drift whatever the published numbers are.
    stale = current.replace(marker, f"| Edited | 0.999 | by hand |\n{marker}")
    assert stale != current
    target.write_text(stale, encoding="utf-8")
    assert generator.main(["--check", *args]) == 1
    assert target.read_text(encoding="utf-8") == stale


def test_the_key_result_carries_k_n_and_both_intervals() -> None:
    """The one number on the README's first screen keeps its interval and its counterweight."""
    generator = _load_generator()
    payload = json.loads((REPO_ROOT / "scorecards/live-v1.1.0.json").read_text(encoding="utf-8"))
    block = generator.render_key_result(payload)
    assert "**Refusal recall 0.722**: 13/18" in block
    assert "Wilson 95 % CI [0.491, 0.875]" in block
    assert "refusal precision 0.929 (13/14, CI [0.685, 0.987])" in block
    assert "](docs/results.md)" in block


def test_the_key_result_warns_when_the_scorecard_is_a_stub() -> None:
    generator = _load_generator()
    payload = json.loads((REPO_ROOT / "scorecards/live-v1.1.0.json").read_text(encoding="utf-8"))
    payload["provenance"]["agent_model"] = "stub-agent"
    assert "Generated on stubs" in generator.render_key_result(payload)


def test_only_the_test_split_is_called_held_out() -> None:
    generator = _load_generator()
    payload = json.loads((REPO_ROOT / "scorecards/live-v1.1.0.json").read_text(encoding="utf-8"))
    assert "held-out `test` split" in generator.render_key_result(payload)
    payload["dataset"]["name"] = "dev"
    block = generator.render_key_result(payload)
    assert "held-out" not in block and "`dev` dataset" in block


def test_a_readme_without_key_result_markers_fails_loudly(tmp_path: Path) -> None:
    generator = _load_generator()
    readme, results = tmp_path / "README.md", tmp_path / "results.md"
    readme.write_text("no markers\n", encoding="utf-8")
    results.write_text("<!-- RESULTS:START -->\n<!-- RESULTS:END -->\n", encoding="utf-8")
    card = tmp_path / "SYSTEM_CARD.md"
    card.write_text("<!-- SYSTEM-CARD:START -->\n<!-- SYSTEM-CARD:END -->\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="KEY-RESULT"):
        generator.main(
            ["--readme", str(readme), "--results-doc", str(results), "--system-card", str(card)]
        )
    assert readme.read_text(encoding="utf-8") == "no markers\n"


def test_results_links_are_relative_to_the_doc_they_sit_in() -> None:
    generator = _load_generator()
    payload = json.loads((REPO_ROOT / "scorecards/live-v1.1.0.json").read_text(encoding="utf-8"))
    assert "](../scorecards/live-v1.1.0.json)" in generator.render_block(payload)


def test_errored_examples_are_counted_in_the_scorecard_and_the_readme_block() -> None:
    """Two of 18 examples errored: they have no audit record."""
    base = _fixture_result()
    scorecard = base.scorecard.model_copy(
        update={"n_examples": 18, "errored_example_ids": ("q01", "q07")}
    )
    payload = build_headline_dict(
        HarnessResult(**{**base.__dict__, "scorecard": scorecard})  # type: ignore[arg-type]
    )
    assert payload["sample_sizes"]["n_errored"] == 2
    block = _load_generator().render_block(payload)
    assert "**2 errored, 0 not run**" in block


def test_check_fails_when_the_scorecard_differs_from_its_artefact(tmp_path: Path) -> None:
    """A number edited in the scorecard by hand no longer matches the recomputation."""
    generator = _load_generator()
    scorecard = tmp_path / "live.json"
    payload = json.loads((REPO_ROOT / "scorecards/live-v1.0.0.json").read_text(encoding="utf-8"))
    readme, results = tmp_path / "README.md", tmp_path / "results.md"
    readme.write_text("<!-- KEY-RESULT:START -->\n<!-- KEY-RESULT:END -->\n", encoding="utf-8")
    results.write_text("<!-- RESULTS:START -->\n<!-- RESULTS:END -->\n", encoding="utf-8")
    # A card in tmp_path too: the default one is the committed docs/SYSTEM_CARD.md.
    card = tmp_path / "SYSTEM_CARD.md"
    card.write_text("<!-- SYSTEM-CARD:START -->\n<!-- SYSTEM-CARD:END -->\n", encoding="utf-8")
    scorecard.write_text(json.dumps(payload), encoding="utf-8")
    args = ["--scorecard", str(scorecard), "--readme", str(readme), "--results-doc", str(results)]
    args += ["--system-card", str(card)]
    assert generator.main(args) == 0
    assert generator.main(["--check", *args]) == 0

    payload["answer_quality"]["citation_accuracy"]["micro"]["value"] = 0.9
    scorecard.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert generator.main(["--check", *args]) == 1


def test_refusals_are_counted_by_recorded_cause() -> None:
    """The scorecard says which check decided each refusal, split by label, so the
    README can show from the artefact whether the threshold decided any of them."""
    from verity.types import RefusalReason

    base = _fixture_result()
    reasons = {
        11: RefusalReason.NO_CITATION,
        12: RefusalReason.NO_CITATION,
        13: RefusalReason.MODEL_REFUSED,
        14: RefusalReason.BELOW_THRESHOLD,
    }
    audits = [
        a.model_copy(update={"refusal_reason": reasons.get(i)})
        for i, a in enumerate(base.per_audit)
    ]
    result = HarnessResult(**{**base.__dict__, "per_audit": audits})  # type: ignore[arg-type]
    by_reason = build_headline_dict(result)["refusal"]["by_reason"]
    assert by_reason == {
        "below_threshold": {"answerable": 0, "unanswerable": 1},
        "model_refused": {"answerable": 0, "unanswerable": 1},
        "no_citation": {"answerable": 0, "unanswerable": 2},
        "not_recorded": {"answerable": 0, "unanswerable": 1},
    }
    block = _load_generator().render_block(build_headline_dict(result))
    assert "Refusals by cause" in block
    assert "`no_citation` 2 (0 answerable)" in block
    assert "`below_threshold` 1 (0 answerable)" in block
    assert "1 refusal(s) with no recorded cause" in block


def test_block_says_when_the_threshold_decided_no_refusal() -> None:
    from verity.types import RefusalReason

    base = _fixture_result()
    audits = [
        a.model_copy(update={"refusal_reason": RefusalReason.LOW_CITATION_COVERAGE})
        if a.refused
        else a
        for a in base.per_audit
    ]
    result = HarnessResult(**{**base.__dict__, "per_audit": audits})  # type: ignore[arg-type]
    block = _load_generator().render_block(build_headline_dict(result))
    assert "the confidence threshold decided none of them" in block


def test_every_proportion_carries_its_count_and_wilson_interval() -> None:
    payload = build_headline_dict(_fixture_result())
    aq, refu = payload["answer_quality"], payload["refusal"]
    hall = aq["hallucination_rate"]
    assert (hall["k"], hall["n_questions"]) == (1, 11)
    assert hall["ci95_wilson"] == pytest.approx([0.0162, 0.3774], abs=1e-4)
    micro = aq["faithfulness"]["micro"]
    assert (micro["k"], micro["n_claims"]) == (517, 550)
    assert micro["ci95_wilson"][0] < micro["value"] < micro["ci95_wilson"][1]
    # A mean of per-answer ratios is not a proportion: no Wilson interval.
    assert "ci95_wilson" not in aq["faithfulness"]["macro"]
    precision = refu["refusal_precision"]
    assert (precision["k"], precision["n"], precision["value"]) == (5, 5, 1.0)
    assert precision["ci95_wilson"] == pytest.approx([0.5655, 1.0], abs=1e-4)
    assert refu["refusal_recall"]["n"] == 5
    assert refu["out_of_scope_answered"]["ci95_wilson"] == pytest.approx([0.0, 0.4345], abs=1e-4)


def test_block_prints_the_interval_and_n_next_to_each_proportion() -> None:
    block = _load_generator().render_block(build_headline_dict(_fixture_result()))
    rows = {
        line.split("|")[1].strip(" *"): line for line in block.splitlines() if line.startswith("|")
    }
    assert "5/5 · 95 % CI [0.566, 1.000]" in rows["Refusal precision"]
    assert "1/11 · 95 % CI [0.016, 0.377]" in rows["Hallucination rate, answerable only"]
    assert "517/550 · 95 % CI [" in rows["Faithfulness, per claim"]
    assert "CI" not in rows["Faithfulness, per question"]
    assert "0/5 · 95 % CI [0.000, 0.434]" in rows["Unanswerable answered"]


def test_retrieval_rows_do_not_claim_to_be_deterministic() -> None:
    """Retrieval is scored on the agent's hits, which follow the LLM's sub-questions:
    reruns of one commit move nDCG (datasets/eval/runs/noise/)."""
    generator = _load_generator()
    payload = json.loads(generator.DEFAULT_SCORECARD.read_text())
    rows = [line for line in generator.render_block(payload).splitlines() if "| Retrieval" in line]
    assert rows
    assert not any("deterministic" in row for row in rows)
    assert "LLM sub-questions" in rows[0]


# --- system card block ------------------------------------------------------------------


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
    assert rows["Hallucination rate, answerable only ↓ (judge)"] == [
        "0.000",
        "0/11",
        "[0.000, 0.259]",
    ]
    assert rows["Citation accuracy per claim (judge)"] == ["0.618", "21/34", "[0.450, 0.761]"]
    # A mean of per-question fractions has no k: n is shown, no interval is invented.
    assert rows["Retrieval nDCG@8, per-question mean"] == ["0.953", "n=11", "no interval"]


def test_system_card_slos_are_approved_and_say_what_was_not_measured() -> None:
    rows = _card_rows(_committed_scorecard())
    slos = [_row_starting(rows, prefix) for prefix in ("p95", "Error", "Cost")]
    assert all(cells[0].endswith("(approved)") for cells in slos)
    # 0 errors in 16: no violation observed, but the upper bound is far above 1 %.
    assert slos[1][1:] == [
        "0.000 (0/16, 95 % CI [0.000, 0.194])",
        "no violation observed; n too small to conclude (95 % upper bound 19.4% > 1%)",
    ]
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


def test_an_errored_request_is_published_as_a_missed_error_slo() -> None:
    payload = _committed_scorecard()
    payload["sample_sizes"]["n_errored"] = 1
    assert _row_starting(_card_rows(payload), "Error")[1:] == [
        "0.062 (1/16, 95 % CI [0.011, 0.283])",
        "**SLO not met**: 1 of 16 requests errored; the scorecard does not record why",
    ]


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


def test_a_scorecard_older_than_v6_is_refused_by_the_card() -> None:
    """Only v6 stores every k/n the card prints; an older scorecard would need them guessed."""
    payload = _committed_scorecard()
    payload["schema"] = "verity.scorecard.headline/v3"
    with pytest.raises(SystemExit, match="v6"):
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
    stale = regenerated.replace("| 13/18 |", "| 18/18 |")
    assert stale != regenerated
    card.write_text(stale, encoding="utf-8")
    assert generator.main(args) == 1
    assert card.read_text(encoding="utf-8") == stale


def test_invented_answer_rate_counts_every_judged_answer() -> None:
    """Every judged answer is in the denominator, answerable or not."""
    payload = build_headline_dict(_fixture_result())
    invented = payload["answer_quality"]["invented_answer_rate"]
    assert invented["scope"] == "answered"
    # The fixture's 11 judged answers each carry 3 unsupported claims.
    assert (invented["k"], invented["n_questions"]) == (11, 11)
    assert invented["ci95_wilson"] is not None


def test_the_key_result_leads_with_invented_answers_and_names_them() -> None:
    """The README's first number counts unanswerable answers too, with its definition."""
    generator = _load_generator()
    payload = json.loads((REPO_ROOT / "scorecards/live-v1.1.0.json").read_text(encoding="utf-8"))
    block = generator.render_key_result(payload)
    assert block.startswith("**Invented answers 1/40**")
    assert "answerable or not" in block
    assert "`q-089`" in block
    assert "**Refusal recall" not in block.splitlines()[0]


def test_a_zero_error_count_too_small_for_the_target_is_not_met() -> None:
    """0/56 has a Wilson upper bound of 6.4 %: no violation observed, but n cannot show
    a rate of 1 % or less, so the status is not "met"."""
    payload = json.loads((REPO_ROOT / "scorecards/live-v1.1.0.json").read_text(encoding="utf-8"))
    rows = _card_rows(payload)
    error = _row_starting(rows, "Error rate")
    assert error[2].startswith("no violation observed; n too small to conclude")
    assert error[2] != "met"


def test_cost_per_query_is_published_with_and_without_the_judge() -> None:
    generator = _load_generator()
    payload = json.loads((REPO_ROOT / "scorecards/live-v1.1.0.json").read_text(encoding="utf-8"))
    # 54 of the 56 questions were asked before the run's spending cap stopped it.
    assert payload["cost"]["per_query_with_judge_usd"] == pytest.approx(2.4562 / 54, abs=1e-4)
    block = generator.render_block(payload)
    row = next(line for line in block.splitlines() if line.startswith("| Cost / query"))
    assert "$0.0235 agent only" in row and "$0.0455 with the judge" in row


def test_on_the_test_run_the_invented_rate_counts_q074_and_the_hallucination_rate_does_not() -> (
    None
):
    """The two denominators differ on real data: 34 judged answers against 31 answerable."""
    from verity.eval.headline import build_headline_dict_from_eval_run
    from verity.eval.run_store import load_run

    run = load_run(REPO_ROOT / "datasets/eval/runs/live/eval_969e1d5_test.json")
    quality = build_headline_dict_from_eval_run(run)["answer_quality"]
    invented, hall = quality["invented_answer_rate"], quality["hallucination_rate"]
    assert (invented["k"], invented["n_questions"], invented["example_ids"]) == (1, 34, ["q-074"])
    assert (hall["k"], hall["n_questions"]) == (0, 31)


@pytest.mark.parametrize(
    ("k", "n", "status"),
    [
        (0, 56, "no violation observed; n too small to conclude (95 % upper bound 6.4% > 1%)"),
        (1, 200, "1 of 200 within target; n too small to conclude (95 % upper bound 2.8% > 1%)"),
        (1, 1000, "met"),
        (5, 100, "**SLO not met**: cause"),
        (0, 0, "not measured"),
    ],
)
def test_a_rate_slo_meets_only_when_its_upper_bound_does(k: int, n: int, status: str) -> None:
    assert _load_generator()._against_proportion(k, n, 0.01, "cause") == status


def test_the_key_result_names_the_validation_of_its_own_judge_prompt() -> None:
    generator = _load_generator()
    payload = json.loads((REPO_ROOT / "scorecards/live-v1.1.0.json").read_text(encoding="utf-8"))
    judge = generator.judge_label(
        payload["provenance"]["judge_model"],
        payload["provenance"]["prompt_sha256"]["judge_faithfulness"],
    )
    block = generator.render_key_result(payload, judge=judge)
    # The run of record is judged by prompt v0.2.0: its κ, never v0.1.0's interval.
    assert "Judge prompt v0.2.0 κ 0.60 vs blind human on 52 of 60 claims" in block
    assert "0.71" not in block
    assert "all 40 shipped answers were judged" in block


def test_the_key_result_of_an_incomplete_run_says_how_many_questions_completed() -> None:
    generator = _load_generator()
    payload = json.loads((REPO_ROOT / "scorecards/live-v1.1.0.json").read_text(encoding="utf-8"))
    block = generator.render_key_result(payload, judge="judge unvalidated")
    assert (
        "n = 54 of 56 completed (2 not run, stopped by the run's spending cap; 0 errored)" in block
    )


def test_questions_a_budget_cap_left_unrun_are_not_request_errors() -> None:
    """eval_513b839_test stopped at its cap after 54 of 56: no request failed."""
    from verity.eval.headline import build_headline_dict_from_eval_run
    from verity.eval.run_store import load_run

    run = load_run(
        REPO_ROOT
        / "datasets/eval/runs/live/eval_513b839_test.incomplete-20261107T122042441670Z.json"
    )
    head = build_headline_dict_from_eval_run(run)
    sizes, cost = head["sample_sizes"], head["cost"]
    assert (sizes["n_errored"], sizes["n_not_run"]) == (0, 2)
    # Per question asked, as the agent-only figure the harness computes over 54.
    assert cost["per_query_with_judge_usd"] == pytest.approx(cost["total_usd"] / 54, abs=1e-6)


def test_the_system_card_counts_only_the_questions_an_incomplete_run_asked() -> None:
    """The latency sample and the source line of eval_513b839_test hold 54 answers."""
    generator = _load_generator()
    payload = json.loads((REPO_ROOT / "scorecards/live-v1.1.0.json").read_text(encoding="utf-8"))
    rows = _card_rows(payload)
    warm = _row_starting(rows, "p95 latency of one answer, warm")
    assert warm[1] == "9857.9 ms (n=54, after one warmup answer)"
    block = generator.render_system_card_block(
        payload, scorecard_path=REPO_ROOT / "scorecards/live-v1.1.0.json"
    )
    assert "n=54 of 56 asked (36 answerable + 18 out-of-scope)" in block
    table = generator.render_block(payload)
    assert "n=54 of 56 asked (36 answerable + 18 unanswerable)" in table


def test_an_incomplete_run_without_unrun_questions_does_not_blame_the_spending_cap() -> None:
    generator = _load_generator()
    payload = json.loads((REPO_ROOT / "scorecards/live-v1.1.0.json").read_text(encoding="utf-8"))
    payload["sample_sizes"] = {**payload["sample_sizes"], "n_not_run": 0, "n_errored": 2}
    assert generator._completed(payload) == "n = 54 of 56 completed (2 errored)"


def _live_and_comparison() -> tuple[dict, dict]:  # type: ignore[type-arg]
    live = json.loads((REPO_ROOT / "scorecards/live-v1.1.0.json").read_text(encoding="utf-8"))
    two_arm = json.loads(
        (REPO_ROOT / "scorecards/two-arm-comparison.json").read_text(encoding="utf-8")
    )
    return live, two_arm


def test_the_key_result_leads_with_the_pre_registered_comparison() -> None:
    """The figure the profile cites (4/31 against 12/31) is the README's first line,
    rendered from the two-arm scorecard."""
    live, two_arm = _live_and_comparison()
    block = _load_generator().render_key_result(live, comparison=two_arm)
    first = block.splitlines()[0]
    assert first.startswith("**Unanswerable questions answered: Verity 4/31, naive RAG 12/31**")
    assert "exact McNemar, Holm-adjusted p 0.023" in first
    assert "93 questions (dev + test)" in first
    assert "pre-registered" in first.lower()
    assert block.splitlines()[1].startswith("**Invented answers 1/40**")


def test_the_key_result_says_why_its_refusal_figures_do_not_compare() -> None:
    """13/18 (test, self-abstentions as answers) and 4/31 (dev + test, self-abstentions
    as refusals) are different counts; the block says so and links the rule."""
    live, two_arm = _live_and_comparison()
    last = _load_generator().render_key_result(live, comparison=two_arm).splitlines()[-1]
    assert "4/31 does not compare directly with 13/18 refused or 1/40 invented" in last
    assert "preregistration/two-arm-comparison.md#metrics" in last


def test_a_comparison_of_other_runs_is_left_out_of_the_key_result() -> None:
    """A comparison that does not contain the headline run says nothing about it."""
    live, two_arm = _live_and_comparison()
    for artefact in two_arm["arms"]["verity"]["artefacts"]:
        artefact["path"] = "datasets/eval/runs/live/other.json"
    block = _load_generator().render_key_result(live, comparison=two_arm)
    assert block.startswith("**Invented answers")
    assert "4/31" not in block
