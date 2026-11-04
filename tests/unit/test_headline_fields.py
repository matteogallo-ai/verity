"""Headline scorecard: provenance summaries and the stub-over-live safeguard of the file
writer. Written against the mutants mutation testing found alive
(docs/quality/mutation.md)."""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import pytest

from tests.unit.test_headline_generator import _fixture_result
from verity.eval.harness import HarnessResult
from verity.eval.headline import (
    StubOverwritesLiveScorecardError,
    _summarize_stamps,
    write_headline_scorecard,
)
from verity.types import EvalRun


@pytest.mark.parametrize(
    ("models", "total", "expected"),
    [
        ([], 0, "unknown"),
        ([None, ""], 2, "unknown"),
        (["m"], 1, "m (1/1)"),
        (["m", "n", "m"], 3, "m (2/3), n (1/3)"),
        (["m", None], 2, "m (1/2 answered — 1 refusals bypass the judge)"),
    ],
)
def test_stamps_count_only_named_models_and_name_the_bypassed_ones(
    models: list[str | None], total: int, expected: str
) -> None:
    assert _summarize_stamps(models, total_examples=total) == expected


def _run(result: HarnessResult) -> EvalRun:
    return result.as_eval_run()


def _stub(result: HarnessResult, *, only: str | None = None) -> HarnessResult:
    if only == "judge":
        return dataclasses.replace(result, judge_model="stub-judge-v1")
    if only == "agent":
        # A mixed-case agent label alone still marks the run as a stub.
        return dataclasses.replace(result, agent_model="Stub-Agent")
    return dataclasses.replace(result, agent_model="stub-agent", judge_model="stub-judge-v1")


@pytest.mark.parametrize("only", [None, "judge", "agent"])
def test_a_stub_run_cannot_overwrite_a_live_scorecard(tmp_path: Path, only: str | None) -> None:
    path = write_headline_scorecard(_run(_fixture_result()), tmp_path / "live.json")
    before = path.read_text(encoding="utf-8")
    prov = json.loads(before)["provenance"]
    message = (
        "refusing to overwrite live scorecard at live.json with a stub run "
        f"(existing agent={prov['agent_model']!r}, judge={prov['judge_model']!r}). "
        "Pass allow_overwrite=True if this is intentional."
    )
    with pytest.raises(StubOverwritesLiveScorecardError, match=f"^{re.escape(message)}$"):
        write_headline_scorecard(_run(_stub(_fixture_result(), only=only)), path)
    assert path.read_text(encoding="utf-8") == before


def test_a_scorecard_without_a_provenance_object_counts_as_live(tmp_path: Path) -> None:
    path = tmp_path / "live.json"
    path.write_text(json.dumps({"provenance": "claude"}), encoding="utf-8")
    with pytest.raises(StubOverwritesLiveScorecardError):
        write_headline_scorecard(_run(_stub(_fixture_result())), path)


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
    write_headline_scorecard(_run(_stub(_fixture_result())), path)
    agent = json.loads(path.read_text(encoding="utf-8"))["provenance"]["agent_model"]
    assert agent.startswith("claude-sonnet-4-6")


def test_a_live_run_and_an_explicit_override_may_replace_a_live_scorecard(
    tmp_path: Path,
) -> None:
    path = write_headline_scorecard(_run(_fixture_result()), tmp_path / "live.json")
    write_headline_scorecard(_run(_fixture_result()), path)
    write_headline_scorecard(_run(_stub(_fixture_result())), path, allow_overwrite=True)
