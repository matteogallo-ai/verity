"""Enriched scorecard JSON — the file the README's headline table renders from.

Two audiences:

- The **README generator** at ``scripts/generate_headline_results.py`` reads
  this JSON and emits the "measured results" section between marker comments
  in the root ``README.md``. Numbers are copied verbatim, no rescaling.
- Reviewers who want a machine-readable v1.0.0 artefact without walking the
  whole ``EvalRun`` structure. Every metric on this file carries the
  provenance triple + total_cost_usd + timestamp so a chiffre can never be
  quoted without knowing the run that produced it.

Provenance aggregation is deliberately per-request, not boot-config: the
strings ``agent_model`` / ``judge_model`` in this JSON are summaries computed
from the per-answer / per-judge-call stamps that the router actually served.
This carries the S6/S7 per-request-provenance invariant into the reporting
layer — a fallback that swapped provider mid-run shows up as an honest split
("anthropic 15/16, openai 1/16") instead of the boot-time first-client label.

Schema v3 (pre-tag correction, 2026-09-07): every judge-track metric carries
its **honest denominator** and **scope description**. Refusals bypass the
judge (no LLM call happens), and the previous v2 aggregate averaged them in
with a convention value of 1.0 — inflating ``citation_accuracy`` in
particular. v3 reports each metric only over its meaningful sample:

- ``faithfulness`` and ``citation_accuracy``: scope = ``answered`` (the
  ``n_judged`` answers the judge actually saw).
- ``hallucination_rate``: scope = ``answerable_and_answered`` (the at-risk
  set — the only questions where the metric formula can return 1.0).

The judge provenance summary now reads e.g.
``"claude-sonnet-4-6 (12/16 answered — 4 refusals bypass the judge)"`` when
refusals cause skips, so the reader can never confuse the judge call count
with the total question count.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity.eval.harness import HarnessResult
from verity.types import EvalRun


def _summarize_stamps(items: list[tuple[str | None, str | None]]) -> str:
    """Aggregate per-call ``(provider, model)`` stamps into a one-line summary.

    Used for the AGENT side, where every dataset example has a stamp (the
    agent runs on every question). See :func:`_summarize_judge_stamps` for the
    judge-side variant that surfaces refusal skips.
    """
    total = sum(1 for _, model in items if isinstance(model, str) and model)
    if total == 0:
        return "unknown"
    counts = Counter(model for _, model in items if isinstance(model, str) and model)
    parts = [f"{model} ({n}/{total})" for model, n in counts.most_common()]
    return ", ".join(parts)


def _summarize_judge_stamps(
    items: list[tuple[str | None, str | None]],
    *,
    total_examples: int,
) -> str:
    """Aggregate per-call judge stamps — same as :func:`_summarize_stamps`
    for the "who served" split, but surfaces refusal skips explicitly.

    Format when the judge ran on every example (no refusals):
        ``"<model> (N/N)"``
    Format when refusals bypassed the judge on some examples:
        ``"<model> (K/N answered — <skipped> refusals bypass the judge)"``

    Load-bearing honesty rule: a reader must not be able to confuse the
    judge call count with the total question count. The v2 output
    ``"(12/12)"`` was arithmetically true (12 non-None stamps out of 12
    counted stamps) but misleading — 4 refusals had already been filtered
    out. v3 puts the total in the denominator and names the skips.
    """
    counted = sum(1 for _, model in items if isinstance(model, str) and model)
    if counted == 0:
        return "unknown"
    counts = Counter(model for _, model in items if isinstance(model, str) and model)
    skipped = total_examples - counted
    if skipped <= 0:
        parts = [f"{model} ({n}/{counted})" for model, n in counts.most_common()]
        return ", ".join(parts)
    # Skipped calls exist — surface them.
    parts = [
        f"{model} ({n}/{total_examples} answered — {skipped} refusals bypass the judge)"
        for model, n in counts.most_common()
    ]
    return ", ".join(parts)


def _hash_dataset_path(dataset_path: Path | None) -> str | None:
    """SHA-256 of the dataset file's bytes (or ``None`` if not available)."""
    if dataset_path is None or not dataset_path.is_file():
        return None
    h = hashlib.sha256()
    h.update(dataset_path.read_bytes())
    return h.hexdigest()


@dataclass(frozen=True)
class _NormalizedInputs:
    """The minimal set of inputs ``_compute_headline`` needs.

    Serves two callers: :func:`build_headline_dict` (fresh from harness) and
    :func:`build_headline_dict_from_eval_run` (rehydrated from a persisted
    :class:`EvalRun`, e.g. to regenerate the scorecard from an audit trail
    without re-spending on the judge).
    """

    scorecard_git_sha: str
    scorecard_created_at_iso: str
    scorecard_dataset: str
    n_examples: int
    retrieval_k: int
    retrieval_precision: float
    retrieval_recall: float
    retrieval_ndcg: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    cold_start_ms: float | None
    cost_per_query_usd: float
    agent_cost_usd: float
    judge_cost_usd: float
    embedding_model: str
    agent_summary: str
    judge_summary: str
    per_refusal_records: list[dict[str, Any]]
    per_audit_records: list[
        dict[str, Any]
    ]  # {expected_answerable, refused, judge_claims_is_none, faithfulness, citation_accuracy, hallucination_rate}


def _compute_headline(inputs: _NormalizedInputs, *, dataset_sha: str | None) -> dict[str, Any]:
    """Produce the v3 dict. Every scoped metric is derived from ``per_audit``
    with its explicit denominator — no metric is averaged over 16 with
    refusals-as-1.0 gratuitously boosting it."""
    n_answerable = sum(1 for r in inputs.per_refusal_records if r["expected_answerable"])
    n_unanswerable = sum(1 for r in inputs.per_refusal_records if not r["expected_answerable"])
    n_answered = sum(1 for a in inputs.per_audit_records if not a["refused"])
    n_judged = sum(1 for a in inputs.per_audit_records if not a["judge_claims_is_none"])
    n_answerable_answered = sum(
        1 for a in inputs.per_audit_records if a["expected_answerable"] and not a["refused"]
    )
    n_refused = sum(1 for a in inputs.per_audit_records if a["refused"])

    # Faithfulness + citation_accuracy: scope = answered (the judge actually saw
    # these). Refusals contribute 1.0 by judge convention in the per-example
    # metric, but that convention would inflate the aggregate when averaged over
    # all 16, so we scope to the n_judged subset.
    judged = [a for a in inputs.per_audit_records if not a["judge_claims_is_none"]]
    faith_over_answered = sum(a["faithfulness"] for a in judged) / len(judged) if judged else 0.0
    ca_over_answered = sum(a["citation_accuracy"] for a in judged) / len(judged) if judged else 0.0

    # Hallucination rate: scope = "answerable AND answered" (the at-risk set —
    # the only questions where the per-example formula
    # ``1.0 if (example.answerable and supported < total) else 0.0`` can return 1.0).
    at_risk = [a for a in inputs.per_audit_records if a["expected_answerable"] and not a["refused"]]
    hall_over_at_risk = (
        sum(a["hallucination_rate"] for a in at_risk) / len(at_risk) if at_risk else 0.0
    )

    refusal_confusion = _refusal_counts(inputs.per_refusal_records)

    return {
        "schema": "verity.scorecard.headline/v3",
        "generated_at": inputs.scorecard_created_at_iso,
        "provenance": {
            "run_sha": inputs.scorecard_git_sha,
            "agent_model": inputs.agent_summary,
            "judge_model": inputs.judge_summary,
            "embedding_model": inputs.embedding_model,
        },
        "dataset": {
            "name": inputs.scorecard_dataset,
            "n_questions": inputs.n_examples,
            "n_answerable": n_answerable,
            "n_unanswerable": n_unanswerable,
            "sha256": dataset_sha,
        },
        "sample_sizes": {
            # Explicit sample-size table so every metric's denominator is one
            # click away from the scorecard, not buried in a comment.
            "n_examples": inputs.n_examples,
            "n_answered": n_answered,
            "n_judged": n_judged,
            "n_refused": n_refused,
            "n_answerable_answered": n_answerable_answered,
        },
        "cost": {
            "total_usd": round(inputs.agent_cost_usd + inputs.judge_cost_usd, 6),
            "agent_usd": round(inputs.agent_cost_usd, 6),
            "judge_usd": round(inputs.judge_cost_usd, 6),
            "per_query_usd": round(inputs.cost_per_query_usd, 6),
        },
        "retrieval": {
            "k": inputs.retrieval_k,
            "precision_at_k": inputs.retrieval_precision,
            "recall_at_k": inputs.retrieval_recall,
            "ndcg_at_k": inputs.retrieval_ndcg,
        },
        "answer_quality": {
            # v3 rename: each metric is a {value, scope, n_denominator} triple
            # so no chiffre can be quoted without its scope. Refusals no longer
            # boost the aggregate — the denominators shrink to the meaningful
            # sample. See module docstring for the full rationale.
            "faithfulness": {
                "value": round(faith_over_answered, 6),
                "scope": "answered",
                "n_denominator": n_judged,
            },
            "citation_accuracy": {
                "value": round(ca_over_answered, 6),
                "scope": "answered",
                "n_denominator": n_judged,
            },
            "hallucination_rate": {
                "value": round(hall_over_at_risk, 6),
                "scope": "answerable_and_answered",
                "n_denominator": n_answerable_answered,
            },
        },
        "refusal_calibration": {
            "refusal_precision": refusal_confusion["precision"],
            "refusal_recall": refusal_confusion["recall"],
            "out_of_scope_answered": {
                "count": refusal_confusion["fn"],
                "total": n_unanswerable,
            },
        },
        "latency_ms": {
            "p50": inputs.latency_p50,
            "p95": inputs.latency_p95,
            "p99": inputs.latency_p99,
            "cold_start": inputs.cold_start_ms,
        },
        "per_refusal": inputs.per_refusal_records,
    }


def _refusal_counts(per_refusal_records: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(
        1 for r in per_refusal_records if r["observed_refused"] and not r["expected_answerable"]
    )
    fp = sum(1 for r in per_refusal_records if r["observed_refused"] and r["expected_answerable"])
    fn = sum(
        1 for r in per_refusal_records if not r["observed_refused"] and not r["expected_answerable"]
    )
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall}


def build_headline_dict(
    result: HarnessResult,
    *,
    dataset_path: Path | None = None,
) -> dict[str, Any]:
    """Assemble the enriched headline JSON dict from a fresh ``HarnessResult``.

    See module docstring for the v3 scoping rules.
    """
    sc = result.scorecard
    agent_cost_usd = sum(a.usage.cost_usd for a in result.answers)

    agent_stamps: list[tuple[str | None, str | None]] = [
        (a.provider_used.value if a.provider_used is not None else None, a.model_used)
        for a in result.answers
    ]
    agent_summary = _summarize_stamps(agent_stamps)

    judge_stamps = list(getattr(result, "judge_stamps", []))
    if judge_stamps:
        judge_items: list[tuple[str | None, str | None]] = [
            (s.provider.value if s.provider is not None else None, s.model) for s in judge_stamps
        ]
        judge_summary = _summarize_judge_stamps(judge_items, total_examples=sc.n_examples)
    else:
        judge_summary = result.judge_model  # boot-time fallback

    per_refusal_records = [
        {
            "example_id": p.example_id,
            "expected_answerable": p.expected_answerable,
            "observed_refused": p.observed_refused,
        }
        for p in result.per_refusal
    ]
    per_audit_records = [
        {
            "expected_answerable": a.expected_answerable,
            "refused": a.refused,
            "judge_claims_is_none": a.judge_claims is None,
            "faithfulness": a.answer_metrics.faithfulness,
            "citation_accuracy": a.answer_metrics.citation_accuracy,
            "hallucination_rate": a.answer_metrics.hallucination_rate,
        }
        for a in result.per_audit
    ]

    return _compute_headline(
        _NormalizedInputs(
            scorecard_git_sha=sc.git_sha,
            scorecard_created_at_iso=sc.created_at.isoformat(),
            scorecard_dataset=sc.dataset,
            n_examples=sc.n_examples,
            retrieval_k=sc.retrieval.k,
            retrieval_precision=sc.retrieval.precision_at_k,
            retrieval_recall=sc.retrieval.recall_at_k,
            retrieval_ndcg=sc.retrieval.ndcg_at_k,
            latency_p50=sc.latency.p50_ms,
            latency_p95=sc.latency.p95_ms,
            latency_p99=sc.latency.p99_ms,
            cold_start_ms=sc.cold_start_ms,
            cost_per_query_usd=sc.cost_per_query_usd,
            agent_cost_usd=agent_cost_usd,
            judge_cost_usd=result.judge_cost_usd,
            embedding_model=result.embedding_model,
            agent_summary=agent_summary,
            judge_summary=judge_summary,
            per_refusal_records=per_refusal_records,
            per_audit_records=per_audit_records,
        ),
        dataset_sha=_hash_dataset_path(dataset_path),
    )


def build_headline_dict_from_eval_run(
    run: EvalRun,
    *,
    dataset_path: Path | None = None,
    agent_cost_usd: float = 0.0,
    judge_cost_usd: float = 0.0,
) -> dict[str, Any]:
    """Rebuild the headline dict from a persisted :class:`EvalRun` — no
    :class:`HarnessResult` in hand, no LLM re-spend.

    Used when the run of record's audit trail exists on disk but the
    headline schema evolved (e.g. v2 → v3 rename). ``agent_cost_usd`` and
    ``judge_cost_usd`` must be provided by the caller since the persisted
    ``EvalRun`` only carries the ``cost_per_query`` aggregate; if omitted,
    the cost breakdown fields render as ``0.0`` (the ``total_usd`` in the
    scorecard's own ``cost_per_query * n_examples`` still holds).

    The agent / judge summary strings fall back to the scalar model labels
    stored on the ``EvalRun`` (per-call provenance is not preserved in
    ``EvalRun``). For the v1.0.0 run of record this is exactly right —
    the run ran on a single provider throughout.
    """
    sc = run.scorecard
    per_audit_records = [
        {
            "expected_answerable": a.expected_answerable,
            "refused": a.refused,
            "judge_claims_is_none": a.judge_claims is None,
            "faithfulness": a.answer_metrics.faithfulness,
            "citation_accuracy": a.answer_metrics.citation_accuracy,
            "hallucination_rate": a.answer_metrics.hallucination_rate,
        }
        for a in run.per_audit
    ]
    per_refusal_records = [
        {
            "example_id": a.example_id,
            "expected_answerable": a.expected_answerable,
            "observed_refused": a.refused,
        }
        for a in run.per_audit
    ]

    # Provenance strings from the scalar EvalRun fields. Format the judge
    # summary with the same "(K/N answered — S refusals bypass the judge)"
    # rule used by the fresh path — reconstruct the counts from per_audit.
    n_examples = sc.n_examples
    n_judged = sum(1 for a in run.per_audit if a.judge_claims is not None)
    skipped = n_examples - n_judged
    if skipped > 0:
        judge_summary = (
            f"{run.judge_model} ({n_judged}/{n_examples} answered "
            f"— {skipped} refusals bypass the judge)"
        )
    else:
        judge_summary = f"{run.judge_model} ({n_judged}/{n_judged})"
    agent_summary = f"{run.agent_model} ({n_examples}/{n_examples})"

    return _compute_headline(
        _NormalizedInputs(
            scorecard_git_sha=sc.git_sha,
            scorecard_created_at_iso=sc.created_at.isoformat(),
            scorecard_dataset=sc.dataset,
            n_examples=sc.n_examples,
            retrieval_k=sc.retrieval.k,
            retrieval_precision=sc.retrieval.precision_at_k,
            retrieval_recall=sc.retrieval.recall_at_k,
            retrieval_ndcg=sc.retrieval.ndcg_at_k,
            latency_p50=sc.latency.p50_ms,
            latency_p95=sc.latency.p95_ms,
            latency_p99=sc.latency.p99_ms,
            cold_start_ms=sc.cold_start_ms,
            cost_per_query_usd=sc.cost_per_query_usd,
            agent_cost_usd=agent_cost_usd,
            judge_cost_usd=judge_cost_usd,
            embedding_model=run.embedding_model,
            agent_summary=agent_summary,
            judge_summary=judge_summary,
            per_refusal_records=per_refusal_records,
            per_audit_records=per_audit_records,
        ),
        dataset_sha=_hash_dataset_path(dataset_path),
    )


class StubOverwritesLiveScorecardError(RuntimeError):
    """Same class of accident as ``StubOverwritesLiveError`` in the run store,
    for the enriched headline scorecard JSON. Prevents a stub-provenance run
    from silently destroying a committed live-run scorecard artefact."""


def write_headline_scorecard(
    result: HarnessResult,
    path: Path,
    *,
    dataset_path: Path | None = None,
    allow_overwrite: bool = False,
) -> Path:
    """Write the enriched JSON to ``path``, creating parent dirs as needed."""
    if path.is_file() and not allow_overwrite and _is_stub_result(result):
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = None
        if isinstance(existing, dict) and not _existing_is_stub(existing):
            raise StubOverwritesLiveScorecardError(
                f"refusing to overwrite live scorecard at {path.name} with a stub run "
                f"(existing agent={existing.get('provenance', {}).get('agent_model')!r}, "
                f"judge={existing.get('provenance', {}).get('judge_model')!r}). "
                "Pass allow_overwrite=True if this is intentional."
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_headline_dict(result, dataset_path=dataset_path)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return path


def _is_stub_result(result: HarnessResult) -> bool:
    return "stub" in result.agent_model.lower() or "stub" in result.judge_model.lower()


def _existing_is_stub(existing: dict[str, Any]) -> bool:
    prov = existing.get("provenance", {})
    if not isinstance(prov, dict):
        return False
    agent = str(prov.get("agent_model", "")).lower()
    judge = str(prov.get("judge_model", "")).lower()
    return "stub" in agent or "stub" in judge


__all__ = [
    "StubOverwritesLiveScorecardError",
    "build_headline_dict",
    "build_headline_dict_from_eval_run",
    "write_headline_scorecard",
]
