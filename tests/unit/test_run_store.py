"""FileRunStore + diff_runs — round-trip persistence and regression detection."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from verity.eval.run_store import FileRunStore, diff_runs
from verity.types import (
    AnswerMetrics,
    EvalRun,
    LatencyMetrics,
    RetrievalMetrics,
    Scorecard,
)


def _run(
    *,
    sha: str,
    faithfulness: float,
    halluc: float,
    refusal_recall: float,
    ndcg: float,
    created_at: datetime | None = None,
    judge_model: str = "stub-judge-v1",
    agent_model: str = "stub-agent",
) -> EvalRun:
    return EvalRun(
        scorecard=Scorecard(
            git_sha=sha,
            created_at=created_at or datetime(2026, 9, 6, 12, 0, tzinfo=UTC),
            dataset="questions",
            n_examples=16,
            retrieval=RetrievalMetrics(precision_at_k=0.5, recall_at_k=0.9, ndcg_at_k=ndcg, k=8),
            answer=AnswerMetrics(
                faithfulness=faithfulness,
                citation_accuracy=1.0,
                hallucination_rate=halluc,
                refusal_precision=1.0,
                refusal_recall=refusal_recall,
            ),
            latency=LatencyMetrics(p50_ms=100.0, p95_ms=250.0, p99_ms=400.0),
            cost_per_query_usd=0.0,
        ),
        embedding_model="bge-small",
        judge_model=judge_model,
        agent_model=agent_model,
    )


def test_round_trip_save_and_history(tmp_path: Path) -> None:
    store = FileRunStore(tmp_path)
    r1 = _run(
        sha="aaa1111",
        faithfulness=0.90,
        halluc=0.10,
        refusal_recall=0.80,
        ndcg=0.85,
        created_at=datetime(2026, 9, 5, 12, 0, tzinfo=UTC),
        agent_model="stub-agent",
    )
    r2 = _run(
        sha="bbb2222",
        faithfulness=0.95,
        halluc=0.05,
        refusal_recall=1.00,
        ndcg=0.88,
        created_at=datetime(2026, 9, 6, 12, 0, tzinfo=UTC),
        agent_model="claude-sonnet-4-6",
    )
    store.save(r1)
    store.save(r2)
    history = store.history("questions")
    assert [r.scorecard.git_sha for r in history] == ["bbb2222", "aaa1111"]
    # Provenance triple round-trips — refusal chiffres can never be read without
    # knowing which agent produced them.
    assert history[0].agent_model == "claude-sonnet-4-6"
    assert history[1].agent_model == "stub-agent"
    assert store.latest("questions") is not None
    assert store.latest("questions").scorecard.git_sha == "bbb2222"  # type: ignore[union-attr]


def test_legacy_run_without_agent_model_deserialises(tmp_path: Path) -> None:
    """Older JSON scorecards (0.5.0 pre-fix) don't have an ``agent_model`` field.
    The reader defaults them to ``"unknown"`` rather than crashing — no chiffre
    lost, but the caveat is visible."""
    import json

    (tmp_path / "eval_legacy_questions.json").write_text(
        json.dumps(
            {
                "embedding_model": "bge-small",
                "judge_model": "stub-judge-v1",
                # deliberately no "agent_model"
                "notes": None,
                "scorecard": {
                    "git_sha": "legacy",
                    "created_at": "2026-09-06T00:00:00+00:00",
                    "dataset": "questions",
                    "n_examples": 16,
                    "retrieval": {
                        "precision_at_k": 0.1,
                        "recall_at_k": 1.0,
                        "ndcg_at_k": 0.9,
                        "k": 8,
                    },
                    "answer": {
                        "faithfulness": 1.0,
                        "citation_accuracy": 1.0,
                        "hallucination_rate": 0.0,
                        "refusal_precision": 1.0,
                        "refusal_recall": 1.0,
                    },
                    "latency": {"p50_ms": 100.0, "p95_ms": 200.0, "p99_ms": 300.0},
                    "cost_per_query_usd": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    store = FileRunStore(tmp_path)
    latest = store.latest("questions")
    assert latest is not None
    assert latest.agent_model == "unknown"


def test_diff_flags_regressions(tmp_path: Path) -> None:
    previous = _run(sha="p", faithfulness=0.95, halluc=0.05, refusal_recall=1.00, ndcg=0.88)
    # A regression: faithfulness drops, hallucination rises, refusal_recall drops, ndcg drops.
    current = _run(sha="c", faithfulness=0.60, halluc=0.30, refusal_recall=0.60, ndcg=0.60)
    deltas = diff_runs(previous, current)
    by_metric = {d.metric: d for d in deltas}
    assert by_metric["faithfulness"].is_regression
    assert by_metric["hallucination_rate"].is_regression
    assert by_metric["refusal_recall"].is_regression
    assert by_metric["ndcg_at_k"].is_regression
    # Direction assertions.
    assert by_metric["faithfulness"].delta == -0.35
    assert by_metric["hallucination_rate"].delta == 0.25


def test_diff_ignores_epsilon_noise(tmp_path: Path) -> None:
    previous = _run(sha="p", faithfulness=0.95, halluc=0.05, refusal_recall=1.00, ndcg=0.88)
    current = _run(
        sha="c",
        faithfulness=0.9500000001,
        halluc=0.05,
        refusal_recall=1.00,
        ndcg=0.88,
    )
    deltas = diff_runs(previous, current, epsilon=1e-6)
    assert not any(d.is_regression for d in deltas)
