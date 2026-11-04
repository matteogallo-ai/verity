"""Headline scorecard round-trip: artefact -> headline dict -> README render.

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
    """Import the CLI script as a module so we can call render_block / render_readme."""
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
    assert payload["schema"] == "verity.scorecard.headline/v5"
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
    assert "answered answerable questions: 1/11" in block
    assert "refusals bypass the judge" in block  # judge summary honesty


def test_write_headline_scorecard_round_trips(tmp_path: Path) -> None:
    result = _fixture_result()
    out = write_headline_scorecard(result.as_eval_run(), tmp_path / "live.json")
    parsed = json.loads(out.read_text(encoding="utf-8"))
    assert parsed["schema"] == "verity.scorecard.headline/v5"
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
    first = generator.render_readme(readme.read_text(encoding="utf-8"), block)
    second = generator.render_readme(first, block)
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


def test_answer_quality_rows_publish_both_aggregations() -> None:
    generator = _load_generator()
    payload = json.loads(Path("scorecards/live-v1.0.0.json").read_text(encoding="utf-8"))
    rows = [line for line in generator.render_block(payload).splitlines() if line.startswith("|")]
    for label in ("Faithfulness", "Citation accuracy"):
        assert any(r.startswith(f"| {label}, per question |") for r in rows)
        assert any(r.startswith(f"| {label}, per claim |") for r in rows)


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
    current = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    # A line inside the generated block: drift whatever the published numbers are.
    stale = current.replace(
        "<!-- RESULTS:END -->", "| Edited | 0.999 | by hand |\n<!-- RESULTS:END -->"
    )
    assert stale != current
    readme.write_text(stale, encoding="utf-8")
    assert generator.main(["--check", "--readme", str(readme)]) == 1
    assert readme.read_text(encoding="utf-8") == stale


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
    assert "**2 errored**" in block


def test_check_fails_when_the_scorecard_differs_from_its_artefact(tmp_path: Path) -> None:
    """A number edited in the scorecard by hand no longer matches the recomputation."""
    generator = _load_generator()
    scorecard = tmp_path / "live.json"
    payload = json.loads((REPO_ROOT / "scorecards/live-v1.0.0.json").read_text(encoding="utf-8"))
    readme = tmp_path / "README.md"
    readme.write_text("<!-- RESULTS:START -->\n<!-- RESULTS:END -->\n", encoding="utf-8")
    scorecard.write_text(json.dumps(payload), encoding="utf-8")
    assert generator.main(["--scorecard", str(scorecard), "--readme", str(readme)]) == 0
    assert generator.main(["--check", "--scorecard", str(scorecard), "--readme", str(readme)]) == 0

    payload["answer_quality"]["citation_accuracy"]["micro"]["value"] = 0.9
    scorecard.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert generator.main(["--check", "--scorecard", str(scorecard), "--readme", str(readme)]) == 1


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
    assert "1/11 · 95 % CI [0.016, 0.377]" in rows["Hallucination rate"]
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
