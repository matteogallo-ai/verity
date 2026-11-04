"""Persist and diff :class:`EvalRun` records for regression tracking.

One JSON file per (git_sha, dataset) under ``datasets/eval/runs/``
(``eval_<sha>_<dataset>.json``). An incomplete run (budget abort or errored
examples) gets a file of its own, ``eval_<sha>_<dataset>.incomplete-<time>.json``,
which never overwrites anything and which :meth:`FileRunStore.history` skips. Retrieval-only scorecards use
``retrieval_<sha>.json`` in the same directory; the prefixes keep them apart.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from verity.eval.base import RunStore
from verity.types import (
    AnswerMetrics,
    ErroredExample,
    EvalRun,
    IncompleteReason,
    LatencyMetrics,
    PerExampleAudit,
    RetrievalMetrics,
    Scorecard,
)


class StubOverwritesLiveError(RuntimeError):
    """Raised when a stub-provenance run would clobber an on-disk live run
    at the same ``(git_sha, dataset)`` path.

    Guards the audit trail: the same file name is used for stub verification
    runs and live runs of record at a given SHA, so before this guard a
    routine ``verity eval run --ci`` after a $0.30 live run silently
    destroyed the audit data. This error stops that class of accident;
    callers who genuinely want to overwrite pass ``allow_overwrite=True``.
    """


class FileRunStore(RunStore):
    """JSON-file backed :class:`RunStore`. Deterministic filename per SHA + dataset.

    Refuses to clobber a live run with a stub run unless the caller sets
    ``allow_overwrite=True`` on :meth:`save`. See
    :class:`StubOverwritesLiveError`.
    """

    def __init__(self, runs_dir: Path) -> None:
        self._runs_dir = runs_dir

    def save(self, run: EvalRun, *, allow_overwrite: bool = False) -> Path:
        """Write ``run`` and return the path written."""
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        if run.scorecard.incomplete_reason is not None:
            return self._save_incomplete(run)
        path = self.path_for(run.scorecard.git_sha, run.scorecard.dataset)
        if path.is_file() and not allow_overwrite and _is_stub_run(run):
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing = None
            if isinstance(existing, dict) and not _is_stub_payload(existing):
                raise StubOverwritesLiveError(
                    f"refusing to overwrite live run at {path.name} with a stub run "
                    f"(existing agent={existing.get('agent_model')!r}, "
                    f"judge={existing.get('judge_model')!r}). "
                    "Pass allow_overwrite=True if this is intentional."
                )
        payload = _run_to_dict(run)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        return path

    def _save_incomplete(self, run: EvalRun) -> Path:
        # A partial run must never replace a run of record, nor another partial run:
        # each one is spend that cannot be bought back. Mode "x" fails rather than
        # overwrite, and the counter covers two saves within the same microsecond.
        record = self.path_for(run.scorecard.git_sha, run.scorecard.dataset)
        stamp = run.scorecard.created_at.strftime("%Y%m%dT%H%M%S%fZ")
        payload = _run_to_dict(run)
        for n in range(100):
            suffix = f".incomplete-{stamp}" + (f"-{n}" if n else "")
            path = record.with_name(f"{record.stem}{suffix}.json")
            try:
                with path.open("x", encoding="utf-8") as fh:
                    json.dump(payload, fh, indent=2, sort_keys=True)
            except FileExistsError:
                continue
            return path
        raise FileExistsError(f"no free name for an incomplete run next to {record.name}")

    def history(self, dataset: str, limit: int = 50) -> list[EvalRun]:
        if not self._runs_dir.exists():
            return []
        runs: list[EvalRun] = []
        for path in self._runs_dir.iterdir():
            if not path.name.startswith("eval_") or path.suffix != ".json":
                continue
            if ".incomplete-" in path.name:
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

    def path_for(self, git_sha: str, dataset: str) -> Path:
        """Where :meth:`save` writes the run for ``(git_sha, dataset)``."""
        # Dataset in the filename keeps runs on different datasets at the same SHA
        # from overwriting each other (`--dataset` accepts any JSONL file).
        safe = "".join(c for c in dataset if c.isalnum() or c in {"-", "_"}) or "default"
        return self._runs_dir / f"eval_{git_sha}_{safe}.json"


# --------------------------------------------------------------------------------------
# Stub-vs-live provenance detection (audit-trail safeguard)
# --------------------------------------------------------------------------------------


def _is_stub_run(run: EvalRun) -> bool:
    """True iff either the agent or the judge is a stub (canonical labels
    ``stub-agent`` / ``stub-judge-v1``)."""
    return "stub" in run.agent_model.lower() or "stub" in run.judge_model.lower()


def _is_stub_payload(payload: dict[str, object]) -> bool:
    agent = str(payload.get("agent_model", "")).lower()
    judge = str(payload.get("judge_model", "")).lower()
    return "stub" in agent or "stub" in judge


# --------------------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------------------


def _run_to_dict(run: EvalRun) -> dict[str, object]:
    payload: dict[str, object] = {
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
            "cold_start_ms": run.scorecard.cold_start_ms,
        },
    }
    if run.scorecard.errored_example_ids:
        scorecard_payload = payload["scorecard"]
        assert isinstance(scorecard_payload, dict)
        scorecard_payload["errored_example_ids"] = list(run.scorecard.errored_example_ids)
    if run.scorecard.incomplete_reason is not None:
        scorecard_payload = payload["scorecard"]
        assert isinstance(scorecard_payload, dict)
        scorecard_payload["incomplete_reason"] = run.scorecard.incomplete_reason.value
    # Per-example audit trail — omitted from the dict when empty so pre-v1.0.0
    # persisted runs stay byte-identical (regression tests depend on this).
    # ``mode="json"`` serialises UUIDs, tuples etc. into JSON-friendly primitives.
    if run.per_audit:
        payload["per_audit"] = [
            _without_unset_fields(a.model_dump(mode="json")) for a in run.per_audit
        ]
    if run.errored:
        payload["errored"] = [_without_unset_fields(e.model_dump(mode="json")) for e in run.errored]
    # Written only when set, so files from earlier runs keep their exact bytes.
    if run.dataset_sha256 is not None:
        payload["dataset_sha256"] = run.dataset_sha256
    if run.prompt_sha256:
        payload["prompt_sha256"] = dict(sorted(run.prompt_sha256.items()))
    if run.duration_s is not None:
        payload["duration_s"] = run.duration_s
    if run.confidence_threshold is not None:
        payload["confidence_threshold"] = run.confidence_threshold
    return payload


# Audit fields added together: a record from before them has every one of them unset,
# while a later record always has a confidence score.
_PROVENANCE_FIELDS = (
    "confidence_score",
    "refusal_reason",
    "judge_usage",
    "agent_model_used",
    "judge_model_used",
)


def _without_unset_fields(record: dict[str, object]) -> dict[str, object]:
    """Drop ``stage_ms``, the ``retries`` of each usage record, the citation index of each
    judged claim and the provenance fields when they carry nothing.

    Runs recorded before these fields existed then save again byte for byte, so a
    rejudge or a headline regeneration never rewrites a run of record."""
    if all(record.get(key) is None for key in _PROVENANCE_FIELDS):
        for key in _PROVENANCE_FIELDS:
            record.pop(key, None)
    if not record.get("stage_ms"):
        record.pop("stage_ms", None)
    for key in ("usage", "judge_usage"):
        usage = record.get(key)
        if isinstance(usage, dict) and not usage.get("retries"):
            usage.pop("retries", None)
    claims = record.get("judge_claims")
    for claim in claims if isinstance(claims, list) else ():
        if isinstance(claim, dict) and claim.get("citation") is None:
            claim.pop("citation", None)
    return record


def load_run(path: Path) -> EvalRun:
    """The run stored at ``path`` (a file written by :meth:`FileRunStore.save`)."""
    return _run_from_dict(json.loads(path.read_text()))


def _run_from_dict(data: dict[str, object]) -> EvalRun:
    sc = data.get("scorecard")
    assert isinstance(sc, dict)
    created_at = datetime.fromisoformat(str(sc["created_at"]))
    cold_start_raw = sc.get("cold_start_ms")
    scorecard = Scorecard(
        git_sha=str(sc["git_sha"]),
        created_at=created_at,
        dataset=str(sc["dataset"]),
        n_examples=int(sc["n_examples"]),
        retrieval=RetrievalMetrics.model_validate(sc["retrieval"]),
        answer=AnswerMetrics.model_validate(sc["answer"]),
        latency=LatencyMetrics.model_validate(sc["latency"]),
        cost_per_query_usd=float(sc["cost_per_query_usd"]),
        cold_start_ms=float(cold_start_raw) if cold_start_raw is not None else None,
        errored_example_ids=tuple(str(x) for x in sc.get("errored_example_ids") or ()),
        incomplete_reason=(
            IncompleteReason(str(sc["incomplete_reason"])) if sc.get("incomplete_reason") else None
        ),
    )
    per_audit_raw = data.get("per_audit")
    per_audit: tuple[PerExampleAudit, ...]
    if isinstance(per_audit_raw, list):
        per_audit = tuple(PerExampleAudit.model_validate(a) for a in per_audit_raw)
    else:
        per_audit = ()
    errored_raw = data.get("errored")
    errored = (
        tuple(ErroredExample.model_validate(e) for e in errored_raw)
        if isinstance(errored_raw, list)
        else ()
    )
    return EvalRun(
        scorecard=scorecard,
        embedding_model=str(data.get("embedding_model", "")),
        judge_model=str(data.get("judge_model", "")),
        agent_model=str(data.get("agent_model", "unknown")),
        notes=(str(data["notes"]) if data.get("notes") else None),
        per_audit=per_audit,
        dataset_sha256=data.get("dataset_sha256"),
        prompt_sha256=data.get("prompt_sha256") or {},
        duration_s=data.get("duration_s"),
        confidence_threshold=data.get("confidence_threshold"),
        errored=errored,
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
    for ``judge_model`` (a stub-judge number vs a real-judge number for
    faithfulness). ``embedding_model`` isn't checked because it is stable across
    every run so far; if it ever becomes tunable, add it here.
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
