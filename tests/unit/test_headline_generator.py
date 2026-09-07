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
    Provider,
    RetrievalMetrics,
    Scorecard,
    UsageStats,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_headline_results.py"


def _load_generator():  # type: ignore[no-untyped-def]
    """Import the CLI script as a module so we can call render_block / patch_readme."""
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
    )


def test_build_headline_dict_carries_full_provenance_and_totals() -> None:
    result = _fixture_result()
    payload = build_headline_dict(result)
    assert payload["schema"] == "verity.scorecard.headline/v2"
    prov = payload["provenance"]
    # Agent summary: all 16 answers stamped claude-sonnet-4-6 → "claude-sonnet-4-6 (16/16)"
    assert prov["agent_model"] == "claude-sonnet-4-6 (16/16)"
    # Judge summary: 11 answerable questions got judged (5 refusals skipped) → 11/11
    assert prov["judge_model"] == "claude-sonnet-4-6 (11/11)"
    assert prov["embedding_model"] == "BAAI/bge-small-en-v1.5"
    assert prov["run_sha"] == "abc1234"
    # Total cost = 16 answers * $0.05 + judge $0.60 = $1.40 (exact — never rounded away).
    assert payload["cost"]["total_usd"] == pytest.approx(1.4)
    assert payload["cost"]["agent_usd"] == pytest.approx(0.8)
    assert payload["cost"]["judge_usd"] == pytest.approx(0.6)
    # Dataset counts derive from per_refusal — not from raw n_examples.
    assert payload["dataset"]["n_answerable"] == 11
    assert payload["dataset"]["n_unanswerable"] == 5
    # Refusal + faithfulness carried verbatim, no rounding.
    assert payload["refusal_calibration"]["refusal_recall"] == 0.80
    assert payload["answer_quality"]["faithfulness"] == 0.94
    # Renamed field (v2): the ``_answerable`` suffix makes the gating explicit.
    assert payload["answer_quality"]["hallucination_rate_answerable"] == 0.06
    # Companion field (v2): in this fixture per_refusal has i>=11 all refused,
    # so 0 out-of-scope answers were shipped (out of 5). The mixed-provider
    # test below exercises the "1/5" case.
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
    )
    payload = build_headline_dict(result)
    assert payload["provenance"]["agent_model"] == "claude-sonnet-4-6 (15/16), gpt-4.1-mini (1/16)"
    # 11 judge calls, one of them fell back to OpenAI.
    assert payload["provenance"]["judge_model"] == "claude-sonnet-4-6 (10/11), gpt-4.1-mini (1/11)"


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
    assert "0.800" in block  # refusal_recall
    assert "0.940" in block  # faithfulness
    assert "0.950" in block  # ndcg
    assert "claude-sonnet-4-6" in block  # provenance
    assert "abc1234" in block  # commit
    assert "$1.4000" in block  # total cost (4-decimal precision)


def test_write_headline_scorecard_round_trips(tmp_path: Path) -> None:
    result = _fixture_result()
    out = write_headline_scorecard(result, tmp_path / "live.json")
    parsed = json.loads(out.read_text(encoding="utf-8"))
    assert parsed["schema"] == "verity.scorecard.headline/v2"
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
    first = generator.patch_readme(
        readme, generator.render_block(json.loads(scorecard_path.read_text()))
    )
    second = generator.patch_readme(
        readme, generator.render_block(json.loads(scorecard_path.read_text()))
    )
    assert first is True
    assert second is False


def test_generator_fails_loudly_without_markers(tmp_path: Path) -> None:
    result = _fixture_result()
    scorecard_path = tmp_path / "live.json"
    write_headline_scorecard(result, scorecard_path)
    readme = tmp_path / "README.md"
    readme.write_text("no markers here\n", encoding="utf-8")
    generator = _load_generator()
    with pytest.raises(SystemExit):
        generator.patch_readme(
            readme, generator.render_block(json.loads(scorecard_path.read_text()))
        )
