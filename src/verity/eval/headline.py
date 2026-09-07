"""Enriched scorecard JSON — the file the README's headline table renders from.

Two audiences:

- The **README generator** at ``scripts/generate_headline_results.py`` reads
  this JSON and emits the "measured results" section between marker comments
  in the root ``README.md``. Numbers are copied verbatim, no rescaling.
- Reviewers who want a machine-readable v1.0.0 artefact without walking the
  whole ``EvalRun`` structure. Every metric on this file carries the
  provenance triple + total_cost_usd + timestamp so a chiffre can never be
  quoted without knowing the run that produced it.

Written by :func:`write_headline_scorecard` — additive, no logic touched.
Existing ``EvalRun`` JSON under ``datasets/eval/runs/`` stays exactly as it was.

Provenance aggregation is deliberately per-request, not boot-config: the
strings ``agent_model`` / ``judge_model`` in this JSON are summaries computed
from the per-answer / per-judge-call stamps that the router actually served.
This carries the S6/S7 per-request-provenance invariant into the reporting
layer — a fallback that swapped provider mid-run shows up as an honest split
("anthropic 15/16, openai 1/16") instead of the boot-time first-client label.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from verity.eval.harness import HarnessResult


def _summarize_stamps(items: list[tuple[str | None, str | None]]) -> str:
    """Aggregate per-call ``(provider, model)`` stamps into a one-line summary.

    Rules:

    - Only stamps whose ``model`` is a non-empty string count towards the
      total (a ``None`` stamp = "no LLM call happened for this example"; e.g.
      the judge is bypassed on refusals).
    - If every counted call used the same model, the summary is
      ``"<model> (n/N)"``.
    - If multiple models served, they are listed most-common first as
      ``"<model_a> (a/N), <model_b> (b/N)"``.
    - If ``N == 0`` (nothing counted), return ``"unknown"`` — honest fallback.

    The provider field is not shown in the summary string itself because
    ``model`` is the granularity most readers reason about; if a caller needs
    the provider breakdown, they can pull it from the ``per_call`` array
    below.
    """
    total = sum(1 for _, model in items if isinstance(model, str) and model)
    if total == 0:
        return "unknown"
    counts = Counter(model for _, model in items if isinstance(model, str) and model)
    parts = [f"{model} ({n}/{total})" for model, n in counts.most_common()]
    return ", ".join(parts)


def _hash_dataset_path(dataset_path: Path | None) -> str | None:
    """SHA-256 of the dataset file's bytes (or ``None`` if not available)."""
    if dataset_path is None or not dataset_path.is_file():
        return None
    h = hashlib.sha256()
    h.update(dataset_path.read_bytes())
    return h.hexdigest()


def build_headline_dict(
    result: HarnessResult,
    *,
    dataset_path: Path | None = None,
) -> dict[str, Any]:
    """Assemble the enriched headline JSON dict — every field the README needs.

    ``total_cost_usd`` is the exact sum of every observed LLM call (agent +
    judge) over the loop, never a rounded aggregate of ``cost_per_query`` *
    ``n``. ``n_answerable`` and ``n_unanswerable`` are derived from
    :attr:`HarnessResult.per_refusal` so the reader sees at a glance how many
    of each class the run scored on.

    ``provenance.agent_model`` and ``provenance.judge_model`` are aggregated
    from per-request stamps on the Answers / JudgeCallStamps (see module
    docstring). ``dataset.sha256`` locks the dataset identity so a live run
    can be verified as having scored the same file the stub validated.
    """
    sc = result.scorecard
    agent_cost_usd = sum(a.usage.cost_usd for a in result.answers)
    total_cost_usd = agent_cost_usd + result.judge_cost_usd
    n_answerable = sum(1 for p in result.per_refusal if p.expected_answerable)
    n_unanswerable = sum(1 for p in result.per_refusal if not p.expected_answerable)

    # Per-request agent provenance from the Answers.
    agent_stamps: list[tuple[str | None, str | None]] = [
        (a.provider_used.value if a.provider_used is not None else None, a.model_used)
        for a in result.answers
    ]
    agent_summary = _summarize_stamps(agent_stamps)

    # Per-call judge provenance. Callers using an older HarnessResult (no
    # ``judge_stamps`` field) get the legacy scalar model_label as a graceful
    # fallback so this reporting layer stays additive.
    judge_stamps = list(getattr(result, "judge_stamps", []))
    if judge_stamps:
        judge_items: list[tuple[str | None, str | None]] = [
            (s.provider.value if s.provider is not None else None, s.model) for s in judge_stamps
        ]
        judge_summary = _summarize_stamps(judge_items)
    else:
        judge_summary = result.judge_model  # boot-time fallback

    dataset_sha = _hash_dataset_path(dataset_path)

    return {
        # v2 (2026-09-07):
        #   - renamed ``hallucination_rate`` → ``hallucination_rate_answerable``
        #     to make the metric's ``example.answerable``-gating explicit;
        #   - added companion ``out_of_scope_answered`` (count/total) under
        #     ``refusal_calibration`` for the half the answerable-gated
        #     metric cannot see.
        # v1 lives in git history if a reader needs the previous shape.
        "schema": "verity.scorecard.headline/v2",
        "generated_at": sc.created_at.isoformat(),
        "provenance": {
            "run_sha": sc.git_sha,
            "agent_model": agent_summary,
            "judge_model": judge_summary,
            "embedding_model": result.embedding_model,
        },
        "dataset": {
            "name": sc.dataset,
            "n_questions": sc.n_examples,
            "n_answerable": n_answerable,
            "n_unanswerable": n_unanswerable,
            "sha256": dataset_sha,
        },
        "cost": {
            "total_usd": round(total_cost_usd, 6),
            "agent_usd": round(agent_cost_usd, 6),
            "judge_usd": round(result.judge_cost_usd, 6),
            "per_query_usd": round(sc.cost_per_query_usd, 6),
        },
        "retrieval": {
            "k": sc.retrieval.k,
            "precision_at_k": sc.retrieval.precision_at_k,
            "recall_at_k": sc.retrieval.recall_at_k,
            "ndcg_at_k": sc.retrieval.ndcg_at_k,
        },
        "answer_quality": {
            "faithfulness": sc.answer.faithfulness,
            "citation_accuracy": sc.answer.citation_accuracy,
            # Renamed from ``hallucination_rate`` (schema v1) to
            # ``hallucination_rate_answerable`` in v2 for honesty: the metric
            # is gated on ``example.answerable`` in the judge, so it says
            # nothing about hallucination on out-of-scope questions. The
            # companion field ``out_of_scope_answered`` covers that half.
            "hallucination_rate_answerable": sc.answer.hallucination_rate,
        },
        "refusal_calibration": {
            "refusal_precision": sc.answer.refusal_precision,
            "refusal_recall": sc.answer.refusal_recall,
            # Companion to ``hallucination_rate_answerable``: out-of-scope
            # questions the agent did NOT refuse. This is the half the
            # ``answerable``-gated metric cannot see. Count + total so the
            # ratio is unambiguous and the reader knows the sample size.
            "out_of_scope_answered": {
                "count": sum(
                    1
                    for p in result.per_refusal
                    if not p.expected_answerable and not p.observed_refused
                ),
                "total": n_unanswerable,
            },
        },
        "latency_ms": {
            "p50": sc.latency.p50_ms,
            "p95": sc.latency.p95_ms,
            "p99": sc.latency.p99_ms,
            "cold_start": sc.cold_start_ms,
        },
        "per_refusal": [
            {
                "example_id": p.example_id,
                "expected_answerable": p.expected_answerable,
                "observed_refused": p.observed_refused,
            }
            for p in result.per_refusal
        ],
    }


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
    """Write the enriched JSON to ``path``, creating parent dirs as needed.

    Refuses to overwrite an existing live-provenance scorecard on disk when
    the incoming ``result`` is stub-provenance (agent or judge is a stub).
    Pass ``allow_overwrite=True`` to bypass — the safeguard exists to prevent
    the exact class of accident the S7 audit exposed (a routine ``--ci`` run
    clobbering a $0.28 live scorecard)."""
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
    "write_headline_scorecard",
]
