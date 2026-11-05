"""The headline scorecard JSON: the file docs/results.md and the README's key result render from.

It is computed from one :class:`~verity.types.EvalRun` artefact and nothing else
(:func:`build_headline_dict_from_eval_run`), so ``scripts/generate_headline_results.py
--check`` can rebuild it from the artefact and fail on any difference. A fresh run
goes through the same function via :meth:`HarnessResult.as_eval_run`.

Schema v5:

- answer-quality metrics come from :mod:`verity.eval.answer_quality` (the one
  definition): faithfulness and citation accuracy carry a ``macro`` (per question)
  and a ``micro`` (per claim) value, each with its denominator; refusals bypass the
  judge and are in neither;
- ``sample_sizes`` counts judged answers, claims, and judged answers without a usable claim;
- model summaries come from the per-example models recorded in the artefact, so a
  routing fallback shows up as a split, e.g. ``"claude-sonnet-4-6 (15/16),
  gpt-4.1-mini (1/16)"``; the judge summary names the refusals that bypassed it;
- every proportion (per-claim ratios, hallucination rate, refusal precision and
  recall, unanswerable answered) carries its count ``k``, its denominator and its
  95 % Wilson interval ``ci95_wilson``; a per-question mean of ratios is not a
  proportion and has none;
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
from verity.eval.intervals import wilson
from verity.eval.refusal import RefusalOutcome, build_confusion
from verity.types import EvalRun, PerExampleAudit, UsageStats


def _summarize_stamps(models: Sequence[str | None], *, total_examples: int) -> str:
    """Per-model counts of the calls that ran, e.g. ``"claude-sonnet-4-6 (15/16),
    gpt-4.1-mini (1/16)"``; ``None`` entries are examples where no call ran. When
    some did not (refusals bypass the judge), the count reads
    ``"<model> (K/N answered — S refusals bypass the judge)"`` so the number of
    calls is never mistaken for the number of questions."""
    counts = Counter(m for m in models if m)
    counted = sum(counts.values())
    if counted == 0:
        return "unknown"
    skipped = total_examples - counted
    if skipped <= 0:
        return ", ".join(f"{model} ({n}/{counted})" for model, n in counts.most_common())
    return ", ".join(
        f"{model} ({n}/{total_examples} answered — {skipped} refusals bypass the judge)"
        for model, n in counts.most_common()
    )


def _hash_dataset_path(dataset_path: Path | None) -> str | None:
    """SHA-256 of the dataset file's bytes (or ``None`` if not available)."""
    if dataset_path is None or not dataset_path.is_file():
        return None
    h = hashlib.sha256()
    h.update(dataset_path.read_bytes())
    return h.hexdigest()


SCHEMA = "verity.scorecard.headline/v5"


def _interval(k: int, n: int) -> list[float] | None:
    ci = wilson(k, n)
    return None if ci is None else [round(ci[0], 6), round(ci[1], 6)]


def _count(numerator: float) -> int:
    if not float(numerator).is_integer():
        raise ValueError(f"a proportion needs a whole count, got {numerator}")
    return int(numerator)


def _ratio_dict(ratio: Ratio, *, unit: str, proportion: bool) -> dict[str, Any]:
    out: dict[str, Any] = {
        "value": None if ratio.value is None else round(ratio.value, 6),
        f"n_{unit}": ratio.denominator,
    }
    if proportion:
        k = _count(ratio.numerator)
        out["k"] = k
        out["ci95_wilson"] = _interval(k, ratio.denominator)
    return out


def _proportion(k: int, n: int) -> dict[str, Any]:
    return {
        "value": round(k / n, 6) if n else None,
        "k": k,
        "n": n,
        "ci95_wilson": _interval(k, n),
    }


def _refusals_by_reason(audits: Sequence[PerExampleAudit]) -> dict[str, dict[str, int]]:
    """Refusals per recorded cause, split by gold label. Artefacts from before the
    cause was recorded count under ``not_recorded``."""
    counts: dict[str, dict[str, int]] = {}
    for a in audits:
        if not a.refused:
            continue
        row = counts.setdefault(
            str(a.refusal_reason) if a.refusal_reason else "not_recorded",
            {"answerable": 0, "unanswerable": 0},
        )
        row["answerable" if a.expected_answerable else "unanswerable"] += 1
    return dict(sorted(counts.items()))


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
            "agent_model": _summarize_stamps(agent_models, total_examples=n_audited),
            "judge_model": _summarize_stamps(judge_models, total_examples=n_audited),
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
            "n_judged_without_claims": quality.n_judged_without_claims,
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
                "macro": _ratio_dict(
                    quality.faithfulness_macro, unit="questions", proportion=False
                ),
                "micro": _ratio_dict(quality.faithfulness_micro, unit="claims", proportion=True),
            },
            "citation_accuracy": {
                "scope": "answered",
                "macro": _ratio_dict(
                    quality.citation_accuracy_macro, unit="questions", proportion=False
                ),
                "micro": _ratio_dict(
                    quality.citation_accuracy_micro, unit="claims", proportion=True
                ),
            },
            "hallucination_rate": {
                "scope": "answerable_and_answered",
                **_ratio_dict(quality.hallucination_rate, unit="questions", proportion=True),
            },
        },
        "refusal": {
            "confidence_threshold": run.confidence_threshold,
            "refusal_precision": _proportion(tp, tp + fp),
            "refusal_recall": _proportion(tp, tp + fn),
            "out_of_scope_answered": {
                "count": fn,
                "total": n_unanswerable,
                "ci95_wilson": _interval(fn, n_unanswerable),
            },
            "by_reason": _refusals_by_reason(audits),
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
    run: EvalRun,
    path: Path,
    *,
    source_artefact: str | None = None,
    allow_overwrite: bool = False,
) -> Path:
    """Write the headline of ``run`` to ``path``. Pass the saved :class:`EvalRun` and
    its file as ``source_artefact`` so the scorecard is the one
    ``generate_headline_results.py`` rebuilds from that file."""
    if path.is_file() and not allow_overwrite and _is_stub_result(run):
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = None
        if isinstance(existing, dict) and not _existing_is_stub(existing):
            # A provenance that is not an object counts as live; the message must
            # still be built without calling .get on it.
            prov = existing.get("provenance")
            prov = prov if isinstance(prov, dict) else {}
            raise StubOverwritesLiveScorecardError(
                f"refusing to overwrite live scorecard at {path.name} with a stub run "
                f"(existing agent={prov.get('agent_model')!r}, "
                f"judge={prov.get('judge_model')!r}). "
                "Pass allow_overwrite=True if this is intentional."
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_headline_dict_from_eval_run(run, source_artefact=source_artefact)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _is_stub_result(run: EvalRun) -> bool:
    return "stub" in run.agent_model.lower() or "stub" in run.judge_model.lower()


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
