"""Persist and diff :class:`EvalRun` records for regression tracking.

One JSON file per (git_sha, dataset) under ``datasets/eval/runs/`` — see the S2
retrieval scorecards for the sibling format. The naming convention lets both eval
tracks (retrieval-only S2 and full S4) coexist without collision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from verity.eval.base import RunStore
from verity.types import (
    AnswerMetrics,
    EvalRun,
    LatencyMetrics,
    RetrievalMetrics,
    Scorecard,
)


class FileRunStore(RunStore):
    """JSON-file backed :class:`RunStore`. Deterministic filename per SHA + dataset."""

    def __init__(self, runs_dir: Path) -> None:
        self._runs_dir = runs_dir

    def save(self, run: EvalRun) -> None:
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        path = self._path_for(run.scorecard.git_sha, run.scorecard.dataset)
        payload = _run_to_dict(run)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)

    def history(self, dataset: str, limit: int = 50) -> list[EvalRun]:
        if not self._runs_dir.exists():
            return []
        runs: list[EvalRun] = []
        for path in self._runs_dir.iterdir():
            if not path.name.startswith("eval_") or path.suffix != ".json":
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if data.get("scorecard", {}).get("dataset") != dataset:
                continue
            runs.append(_run_from_dict(data))
        runs.sort(key=lambda r: r.scorecard.created_at, reverse=True)
        return runs[:limit]

    def latest(self, dataset: str) -> EvalRun | None:
        h = self.history(dataset, limit=1)
        return h[0] if h else None

    def _path_for(self, git_sha: str, dataset: str) -> Path:
        # Dataset in the filename lets multiple datasets coexist per SHA — cheap
        # future-proofing for when the harness gains a second labeled set.
        safe = "".join(c for c in dataset if c.isalnum() or c in {"-", "_"}) or "default"
        return self._runs_dir / f"eval_{git_sha}_{safe}.json"


# --------------------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------------------


def _run_to_dict(run: EvalRun) -> dict[str, object]:
    return {
        "embedding_model": run.embedding_model,
        "judge_model": run.judge_model,
        "agent_model": run.agent_model,
        "notes": run.notes,
        "scorecard": {
            "git_sha": run.scorecard.git_sha,
            "created_at": run.scorecard.created_at.isoformat(),
            "dataset": run.scorecard.dataset,
            "n_examples": run.scorecard.n_examples,
            "retrieval": run.scorecard.retrieval.model_dump(),
            "answer": run.scorecard.answer.model_dump(),
            "latency": run.scorecard.latency.model_dump(),
            "cost_per_query_usd": run.scorecard.cost_per_query_usd,
        },
    }


def _run_from_dict(data: dict[str, object]) -> EvalRun:
    sc = data.get("scorecard")
    assert isinstance(sc, dict)
    created_at = datetime.fromisoformat(str(sc["created_at"]))
    scorecard = Scorecard(
        git_sha=str(sc["git_sha"]),
        created_at=created_at,
        dataset=str(sc["dataset"]),
        n_examples=int(sc["n_examples"]),
        retrieval=RetrievalMetrics.model_validate(sc["retrieval"]),
        answer=AnswerMetrics.model_validate(sc["answer"]),
        latency=LatencyMetrics.model_validate(sc["latency"]),
        cost_per_query_usd=float(sc["cost_per_query_usd"]),
    )
    return EvalRun(
        scorecard=scorecard,
        embedding_model=str(data.get("embedding_model", "")),
        judge_model=str(data.get("judge_model", "")),
        agent_model=str(data.get("agent_model", "unknown")),
        notes=(str(data["notes"]) if data.get("notes") else None),
    )


# --------------------------------------------------------------------------------------
# Regression diff
# --------------------------------------------------------------------------------------

_HIGHER_IS_BETTER = {
    "precision_at_k",
    "recall_at_k",
    "ndcg_at_k",
    "faithfulness",
    "citation_accuracy",
    "refusal_precision",
    "refusal_recall",
}

_LOWER_IS_BETTER = {
    "hallucination_rate",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "cost_per_query_usd",
}


@dataclass(frozen=True)
class MetricDelta:
    """One metric's diff between two runs — labelled so the CLI can colour it."""

    metric: str
    previous: float
    current: float
    delta: float
    is_regression: bool


def are_compatible(a: EvalRun, b: EvalRun) -> bool:
    """Two runs are comparable iff they share the same dataset AND the same
    provenance triple's *variable* components — the agent model and the judge
    model.

    Comparing a run measured on ``agent=stub-agent`` against one measured on
    ``agent=claude-sonnet-4-6`` is apples-to-oranges: any refusal or latency
    delta reflects the change of agent, not a real regression. Same reasoning
    for ``judge_model`` (a stub-judge chiffre vs a real-judge chiffre for
    faithfulness). ``embedding_model`` isn't checked because it is stable across
    the whole S1-S4 codebase; if it ever becomes tunable, add it here.
    """
    return (
        a.scorecard.dataset == b.scorecard.dataset
        and a.agent_model == b.agent_model
        and a.judge_model == b.judge_model
    )


def diff_runs(previous: EvalRun, current: EvalRun, *, epsilon: float = 1e-6) -> list[MetricDelta]:
    """Compare metric-by-metric. A regression is any delta whose sign is worse than
    the metric's "better" direction, larger than ``epsilon`` in absolute value."""
    deltas: list[MetricDelta] = []
    for name, prev_v, curr_v in _iter_pairs(previous.scorecard, current.scorecard):
        delta = curr_v - prev_v
        is_regression = abs(delta) > epsilon and (
            (name in _HIGHER_IS_BETTER and delta < 0) or (name in _LOWER_IS_BETTER and delta > 0)
        )
        deltas.append(
            MetricDelta(
                metric=name,
                previous=prev_v,
                current=curr_v,
                delta=delta,
                is_regression=is_regression,
            )
        )
    return deltas


def _iter_pairs(prev: Scorecard, curr: Scorecard):  # type: ignore[no-untyped-def]
    for name in ("precision_at_k", "recall_at_k", "ndcg_at_k"):
        yield name, getattr(prev.retrieval, name), getattr(curr.retrieval, name)
    for name in (
        "faithfulness",
        "citation_accuracy",
        "hallucination_rate",
        "refusal_precision",
        "refusal_recall",
    ):
        yield name, getattr(prev.answer, name), getattr(curr.answer, name)
    for name in ("p50_ms", "p95_ms", "p99_ms"):
        yield name, getattr(prev.latency, name), getattr(curr.latency, name)
    yield "cost_per_query_usd", prev.cost_per_query_usd, curr.cost_per_query_usd


__all__ = ["FileRunStore", "MetricDelta", "are_compatible", "diff_runs"]
