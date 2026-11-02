"""The headline scorecard JSON: the file the README's results block renders from.

It is computed from one :class:`~verity.types.EvalRun` artefact and nothing else
(:func:`build_headline_dict_from_eval_run`), so ``scripts/generate_headline_results.py
--check`` can rebuild it from the artefact and fail on any difference. A fresh run
goes through the same function via :meth:`HarnessResult.as_eval_run`.

Schema v4:

- answer-quality metrics come from :mod:`verity.eval.answer_quality` (the one
  definition): faithfulness and citation accuracy carry a ``macro`` (per question)
  and a ``micro`` (per claim) value, each with its denominator; refusals bypass the
  judge and are in neither;
- ``sample_sizes`` counts judged answers, claims and unparseable judge outputs;
- model summaries come from the per-example models recorded in the artefact, so a
  routing fallback shows up as a split, e.g. ``"claude-sonnet-4-6 (15/16),
  gpt-4.1-mini (1/16)"``; the judge summary names the refusals that bypassed it;
- a figure the artefact does not hold is ``null``: refusal precision with no
  refusal, the judge cost of a run recorded before judge usage was stored.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from verity.eval.answer_quality import Ratio, answer_quality
from verity.eval.harness import HarnessResult
from verity.eval.refusal import RefusalOutcome, build_confusion
from verity.types import EvalRun, UsageStats


def _summarize_stamps(models: Sequence[str | None]) -> str:
    """Aggregate per-call ``(provider, model)`` stamps into a one-line summary.

    Used for the AGENT side, where every dataset example has a stamp (the
    agent runs on every question). See :func:`_summarize_judge_stamps` for the
    judge-side variant that surfaces refusal skips.
    """
    counts = Counter(m for m in models if m)
    total = sum(counts.values())
    if total == 0:
        return "unknown"
    parts = [f"{model} ({n}/{total})" for model, n in counts.most_common()]
    return ", ".join(parts)


def _summarize_judge_stamps(
    models: Sequence[str | None],
    *,
    total_examples: int,
) -> str:
    """Aggregate per-call judge stamps — same as :func:`_summarize_stamps`
    for the "who served" split, but surfaces refusal skips explicitly.

    Format when the judge ran on every example (no refusals):
        ``"<model> (N/N)"``
    Format when refusals bypassed the judge on some examples:
        ``"<model> (K/N answered — <skipped> refusals bypass the judge)"``

    Load-bearing rule: a reader must not be able to confuse the
    judge call count with the total question count. The v2 output
    ``"(12/12)"`` was arithmetically true (12 non-None stamps out of 12
    counted stamps) but misleading — 4 refusals had already been filtered
    out. v3 puts the total in the denominator and names the skips.
    """
    counts = Counter(m for m in models if m)
    counted = sum(counts.values())
    if counted == 0:
        return "unknown"
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


SCHEMA = "verity.scorecard.headline/v4"


def _ratio_dict(ratio: Ratio, *, unit: str) -> dict[str, Any]:
    return {
        "value": None if ratio.value is None else round(ratio.value, 6),
        f"n_{unit}": ratio.denominator,
    }


def build_headline_dict_from_eval_run(
    run: EvalRun, *, source_artefact: str | None = None
) -> dict[str, Any]:
    """The headline dict, computed from the run artefact alone (no model call).

    Everything published comes from ``run``: the README generator's ``--check``
    rebuilds this dict from the artefact and compares it with the committed scorecard.
    A figure the artefact does not hold (judge cost and dataset hash of runs recorded
    before they were stored) is ``None``, never a typed-in number.
    """
    sc = run.scorecard
    audits = list(run.per_audit)
    quality = answer_quality(audits)
    n_audited = len(audits)
    n_answerable = sum(1 for a in audits if a.expected_answerable)
    n_unanswerable = n_audited - n_answerable
    n_refused = sum(1 for a in audits if a.refused)
    confusion = build_confusion(
        RefusalOutcome(a.example_id, a.expected_answerable, a.refused) for a in audits
    )
    tp, fp, fn = confusion.tp, confusion.fp, confusion.fn

    # Runs recorded before per-example models were stored fall back to the run labels.
    agent_models = [a.agent_model_used or run.agent_model for a in audits]
    judge_models = [
        (a.judge_model_used or run.judge_model) if a.judge_claims is not None else None
        for a in audits
    ]

    agent_usd = sum(a.usage.cost_usd for a in audits) + run.errored_cost_usd
    judge_recorded = bool(audits) and all(a.judge_usage is not None for a in audits)
    judge_usd = (
        sum(a.judge_usage.cost_usd for a in audits if a.judge_usage) if judge_recorded else None
    )

    def _tokens(usages: list[UsageStats]) -> dict[str, int]:
        return {
            "input": sum(u.input_tokens for u in usages),
            "output": sum(u.output_tokens for u in usages),
        }

    return {
        "schema": SCHEMA,
        "source_artefact": source_artefact,
        "generated_at": sc.created_at.isoformat(),
        "provenance": {
            "run_sha": sc.git_sha,
            "agent_model": _summarize_stamps(agent_models),
            "judge_model": _summarize_judge_stamps(judge_models, total_examples=n_audited),
            "embedding_model": run.embedding_model,
            "prompt_sha256": dict(sorted(run.prompt_sha256.items())),
        },
        "dataset": {
            "name": sc.dataset,
            "n_questions": sc.n_examples,
            "n_answerable": n_answerable,
            "n_unanswerable": n_unanswerable,
            "sha256": run.dataset_sha256,
        },
        "sample_sizes": {
            "n_examples": sc.n_examples,
            # An errored example has no audit record; this holds for runs recorded
            # before errors were listed, which a count of listed ids would not.
            "n_errored": sc.n_examples - n_audited,
            "n_answered": n_audited - n_refused,
            "n_judged": quality.n_judged,
            "n_judge_unparseable": quality.n_judge_unparseable,
            "n_claims": quality.faithfulness_micro.denominator,
            "n_refused": n_refused,
            "n_answerable_answered": quality.hallucination_rate.denominator,
        },
        "cost": {
            "total_usd": None if judge_usd is None else round(agent_usd + judge_usd, 6),
            "agent_usd": round(agent_usd, 6),
            "judge_usd": None if judge_usd is None else round(judge_usd, 6),
            "per_query_usd": round(sc.cost_per_query_usd, 6),
        },
        "run": {
            "duration_s": run.duration_s,
            "agent_tokens": _tokens([a.usage for a in audits]),
            "judge_tokens": (
                _tokens([a.judge_usage for a in audits if a.judge_usage])
                if judge_recorded
                else None
            ),
        },
        "retrieval": {
            "k": sc.retrieval.k,
            "precision_at_k": sc.retrieval.precision_at_k,
            "recall_at_k": sc.retrieval.recall_at_k,
            "ndcg_at_k": sc.retrieval.ndcg_at_k,
        },
        "answer_quality": {
            "faithfulness": {
                "scope": "answered",
                "macro": _ratio_dict(quality.faithfulness_macro, unit="questions"),
                "micro": _ratio_dict(quality.faithfulness_micro, unit="claims"),
            },
            "citation_accuracy": {
                "scope": "answered",
                "macro": _ratio_dict(quality.citation_accuracy_macro, unit="questions"),
                "micro": _ratio_dict(quality.citation_accuracy_micro, unit="claims"),
            },
            "hallucination_rate": {
                "scope": "answerable_and_answered",
                **_ratio_dict(quality.hallucination_rate, unit="questions"),
            },
        },
        "refusal": {
            "confidence_threshold": run.confidence_threshold,
            "refusal_precision": tp / (tp + fp) if tp + fp else None,
            "refusal_recall": tp / (tp + fn) if tp + fn else None,
            "out_of_scope_answered": {"count": fn, "total": n_unanswerable},
        },
        "latency_ms": {
            "p50": sc.latency.p50_ms,
            "p95": sc.latency.p95_ms,
            "p99": sc.latency.p99_ms,
            "cold_start": sc.cold_start_ms,
        },
        "per_refusal": [
            {
                "example_id": a.example_id,
                "expected_answerable": a.expected_answerable,
                "observed_refused": a.refused,
            }
            for a in audits
        ],
    }


def build_headline_dict(
    result: HarnessResult,
    *,
    dataset_path: Path | None = None,
    source_artefact: str | None = None,
) -> dict[str, Any]:
    """The headline of a fresh run: the same computation as from its artefact."""
    run = result.as_eval_run(dataset_sha256=_hash_dataset_path(dataset_path))
    return build_headline_dict_from_eval_run(run, source_artefact=source_artefact)


class StubOverwritesLiveScorecardError(RuntimeError):
    """Same class of accident as ``StubOverwritesLiveError`` in the run store,
    for the enriched headline scorecard JSON. Prevents a stub-provenance run
    from silently destroying a committed live-run scorecard artefact."""


def write_headline_scorecard(
    result: HarnessResult,
    path: Path,
    *,
    dataset_path: Path | None = None,
    source_artefact: str | None = None,
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
    payload = build_headline_dict(
        result, dataset_path=dataset_path, source_artefact=source_artefact
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    "SCHEMA",
    "StubOverwritesLiveScorecardError",
    "build_headline_dict",
    "build_headline_dict_from_eval_run",
    "write_headline_scorecard",
]
