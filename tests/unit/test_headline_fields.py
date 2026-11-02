"""Headline scorecard: every field reaches the JSON, the persisted-run path agrees with
the fresh one, provenance summaries, and the stub-over-live safeguard of the file
writer. Written against the mutants mutation testing found alive
(docs/quality/mutation.md)."""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import pytest

from tests.unit.test_headline_generator import _fixture_result
from verity.eval import build_headline_dict, build_headline_dict_from_eval_run
from verity.eval.harness import HarnessResult
from verity.eval.headline import (
    StubOverwritesLiveScorecardError,
    _summarize_judge_stamps,
    _summarize_stamps,
    write_headline_scorecard,
)
from verity.types import Answer, Confidence, UsageStats


def _all_judged(result: HarnessResult) -> HarnessResult:
    """The fixture with no refusal: the judge ran on every example."""
    audit = [a.model_copy(update={"refused": False, "judge_claims": ()}) for a in result.per_audit]
    refusal = [dataclasses.replace(p, observed_refused=False) for p in result.per_refusal]
    stamps = [
        dataclasses.replace(s, provider=result.answers[0].provider_used, model="claude-sonnet-4-6")
        for s in result.judge_stamps
    ]
    return dataclasses.replace(result, per_audit=audit, per_refusal=refusal, judge_stamps=stamps)


@pytest.mark.parametrize("variant", ["with_refusals", "all_judged"])
def test_a_persisted_run_rebuilds_the_headline_of_the_fresh_result(
    variant: str, tmp_path: Path
) -> None:
    """Single-provider run: the EvalRun path must publish exactly what the fresh path
    publishes, so regenerating a scorecard from the audit trail changes nothing."""
    result = _fixture_result() if variant == "with_refusals" else _all_judged(_fixture_result())
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text('{"id": "q-001"}\n', encoding="utf-8")
    fresh = build_headline_dict(result, dataset_path=dataset)
    rebuilt = build_headline_dict_from_eval_run(
        result.as_eval_run(),
        dataset_path=dataset,
        agent_cost_usd=fresh["cost"]["agent_usd"],
        judge_cost_usd=result.judge_cost_usd,
    )
    assert rebuilt == fresh
    assert fresh["dataset"]["sha256"] is not None


def test_every_scorecard_field_is_copied_verbatim() -> None:
    result = _fixture_result()
    sc = result.scorecard
    payload = build_headline_dict(result)
    assert payload["generated_at"] == sc.created_at.isoformat()
    assert payload["dataset"]["name"] == sc.dataset
    assert payload["dataset"]["n_questions"] == sc.n_examples
    assert payload["retrieval"] == {
        "k": sc.retrieval.k,
        "precision_at_k": sc.retrieval.precision_at_k,
        "recall_at_k": sc.retrieval.recall_at_k,
        "ndcg_at_k": sc.retrieval.ndcg_at_k,
    }
    assert payload["latency_ms"] == {
        "p50": sc.latency.p50_ms,
        "p95": sc.latency.p95_ms,
        "p99": sc.latency.p99_ms,
        "cold_start": sc.cold_start_ms,
    }
    assert payload["per_refusal"] == [
        {
            "example_id": p.example_id,
            "expected_answerable": p.expected_answerable,
            "observed_refused": p.observed_refused,
        }
        for p in result.per_refusal
    ]


def test_cost_fields_are_rounded_to_six_decimals() -> None:
    base = _fixture_result()
    answer = base.answers[0].model_copy(update={"usage": UsageStats(cost_usd=0.12345678)})
    scorecard = base.scorecard.model_copy(update={"cost_per_query_usd": 0.11111111})
    result = dataclasses.replace(
        base, answers=[answer], judge_cost_usd=0.98765432, scorecard=scorecard
    )
    cost = build_headline_dict(result)["cost"]
    assert cost == {
        "agent_usd": 0.123457,
        "judge_usd": 0.987654,
        "per_query_usd": 0.111111,
        "total_usd": 1.111111,
    }


def test_answers_without_provider_and_runs_without_judge_stamps_are_summarised() -> None:
    base = _fixture_result()
    anonymous = Answer(
        query="q",
        text="t",
        confidence=Confidence(score=0.9, refused=False, rationale=""),
        hits_used=(),
        provider_used=None,
        model_used=None,
    )
    result = dataclasses.replace(base, answers=[anonymous], judge_stamps=[])
    provenance = build_headline_dict(result)["provenance"]
    assert provenance["agent_model"] == "unknown"
    assert provenance["judge_model"] == base.judge_model


@pytest.mark.parametrize(
    ("items", "expected"),
    [
        ([], "unknown"),
        ([(None, None), ("anthropic", "")], "unknown"),
        ([("anthropic", "m")], "m (1/1)"),
        ([("anthropic", "m"), ("openai", "n"), ("anthropic", "m")], "m (2/3), n (1/3)"),
    ],
)
def test_agent_stamps_count_only_named_models(
    items: list[tuple[str | None, str | None]], expected: str
) -> None:
    assert _summarize_stamps(items) == expected


@pytest.mark.parametrize(
    ("items", "total", "expected"),
    [
        ([(None, None), ("anthropic", "")], 2, "unknown"),
        ([("anthropic", "m")], 1, "m (1/1)"),
        ([("anthropic", "m"), ("openai", "n")], 2, "m (1/2), n (1/2)"),
        ([("anthropic", "m"), (None, None)], 2, "m (1/2 answered — 1 refusals bypass the judge)"),
    ],
)
def test_judge_stamps_name_the_refusals_that_bypassed_the_judge(
    items: list[tuple[str | None, str | None]], total: int, expected: str
) -> None:
    assert _summarize_judge_stamps(items, total_examples=total) == expected


def test_headline_refusal_pair_is_zero_when_undefined() -> None:
    """Pins the published convention: with no refusal and no out-of-scope question the
    headline prints 0.0 for both, where eval.refusal.ConfusionMatrix uses 1.0."""
    result = _all_judged(_fixture_result())
    result = dataclasses.replace(
        result,
        per_refusal=[dataclasses.replace(p, expected_answerable=True) for p in result.per_refusal],
    )
    calibration = build_headline_dict(result)["refusal_calibration"]
    assert calibration["refusal_precision"] == 0.0
    assert calibration["refusal_recall"] == 0.0


# --- write_headline_scorecard ---------------------------------------------------------


def _stub(result: HarnessResult, *, judge_only: bool = False) -> HarnessResult:
    if judge_only:
        return dataclasses.replace(result, judge_model="stub-judge-v1")
    return dataclasses.replace(result, agent_model="stub-agent", judge_model="stub-judge-v1")


def test_the_file_is_the_sorted_two_space_json_of_the_headline(tmp_path: Path) -> None:
    result = _fixture_result()
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    path = write_headline_scorecard(
        result, tmp_path / "nested" / "dir" / "live.json", dataset_path=dataset
    )
    expected = build_headline_dict(result, dataset_path=dataset)
    assert path.read_text(encoding="utf-8") == json.dumps(expected, indent=2, sort_keys=True)


@pytest.mark.parametrize("judge_only", [False, True])
def test_a_stub_run_cannot_overwrite_a_live_scorecard(tmp_path: Path, judge_only: bool) -> None:
    path = write_headline_scorecard(_fixture_result(), tmp_path / "live.json")
    before = path.read_text(encoding="utf-8")
    message = (
        "refusing to overwrite live scorecard at live.json with a stub run "
        "(existing agent='claude-sonnet-4-6 (16/16)', "
        "judge='claude-sonnet-4-6 (11/16 answered — 5 refusals bypass the judge)'). "
        "Pass allow_overwrite=True if this is intentional."
    )
    with pytest.raises(StubOverwritesLiveScorecardError, match=f"^{re.escape(message)}$"):
        write_headline_scorecard(_stub(_fixture_result(), judge_only=judge_only), path)
    assert path.read_text(encoding="utf-8") == before


def test_a_scorecard_without_a_provenance_object_counts_as_live(tmp_path: Path) -> None:
    path = tmp_path / "live.json"
    path.write_text(json.dumps({"provenance": "claude"}), encoding="utf-8")
    with pytest.raises(StubOverwritesLiveScorecardError):
        write_headline_scorecard(_stub(_fixture_result()), path)


@pytest.mark.parametrize(
    "existing",
    [
        json.dumps({"provenance": {"agent_model": "stub-agent (16/16)", "judge_model": "x"}}),
        json.dumps({"provenance": {"agent_model": "x", "judge_model": "Stub-Judge-v1"}}),
        "not json {",
        json.dumps(["a list, not a scorecard"]),
    ],
    ids=["stub-agent", "stub-judge", "corrupt", "not-an-object"],
)
def test_a_stub_run_may_replace_a_stub_or_unreadable_scorecard(
    tmp_path: Path, existing: str
) -> None:
    path = tmp_path / "scorecard.json"
    path.write_text(existing, encoding="utf-8")
    write_headline_scorecard(_stub(_fixture_result()), path)
    assert json.loads(path.read_text(encoding="utf-8"))["provenance"]["agent_model"].startswith(
        "claude-sonnet-4-6"
    )


def test_a_live_run_and_an_explicit_override_may_replace_a_live_scorecard(
    tmp_path: Path,
) -> None:
    path = write_headline_scorecard(_fixture_result(), tmp_path / "live.json")
    write_headline_scorecard(_fixture_result(), path)
    write_headline_scorecard(_stub(_fixture_result()), path, allow_overwrite=True)
