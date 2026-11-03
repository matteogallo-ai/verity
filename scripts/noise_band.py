"""Run-to-run noise: the min-max range of each metric over repeated live runs.

    uv run python scripts/noise_band.py          # write the summary and the doc block
    uv run python scripts/noise_band.py --check  # exit 1 if either is stale

The runs are the dev run of record (``runs/live/eval_6ec544d_dev.json``) and reruns of
the same commit, dataset, prompts and models (``runs/noise/``, pinned in that
directory's ``SHA256SUMS``). A rerun is not a run of record: the published figures stay
those of the first run; the range only says how far a figure moves when nothing but the
models' sampling changes. With three runs it is a range, not an interval, and a lower
bound on the spread.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
# The marker splicer is shared with the judge-validation block rather than copied.
sys.path.insert(0, str(SCRIPTS))
from render_judge_validation import replace_block  # noqa: E402

REPO_ROOT = SCRIPTS.parent
RUNS = REPO_ROOT / "datasets" / "eval" / "runs"
DEFAULT_RUNS = (
    RUNS / "live" / "eval_6ec544d_dev.json",
    RUNS / "noise" / "eval_6ec544d_dev.rerun2.json",
    RUNS / "noise" / "eval_6ec544d_dev.rerun3.json",
)
SUMMARY = RUNS / "noise" / "noise_band_6ec544d_dev.json"
DOC = REPO_ROOT / "docs" / "evaluation-methodology.md"
START, END = "<!-- NOISE:START -->", "<!-- NOISE:END -->"

# Everything that must be identical for the spread to be noise and not a change.
SAME = (
    "agent_model",
    "judge_model",
    "embedding_model",
    "dataset_sha256",
    "prompt_sha256",
    "confidence_threshold",
)
# (label, path in the scorecard, format)
METRICS = (
    ("Faithfulness, per-question mean (judge)", ("answer", "faithfulness"), "{:.3f}"),
    ("Citation accuracy, per-question mean (judge)", ("answer", "citation_accuracy"), "{:.3f}"),
    ("Hallucination rate (judge)", ("answer", "hallucination_rate"), "{:.3f}"),
    ("Refusal precision", ("answer", "refusal_precision"), "{:.3f}"),
    ("Refusal recall", ("answer", "refusal_recall"), "{:.3f}"),
    ("Retrieval nDCG@8", ("retrieval", "ndcg_at_k"), "{:.3f}"),
    ("Retrieval recall@8", ("retrieval", "recall_at_k"), "{:.3f}"),
    ("Retrieval precision@8", ("retrieval", "precision_at_k"), "{:.3f}"),
    ("Latency p50 (ms, warm)", ("latency", "p50_ms"), "{:.0f}"),
    ("Latency p95 (ms, warm)", ("latency", "p95_ms"), "{:.0f}"),
    ("Latency p99 (ms, warm)", ("latency", "p99_ms"), "{:.0f}"),
    ("Cold start (ms)", ("cold_start_ms",), "{:.0f}"),
    ("Cost per query (USD, agent)", ("cost_per_query_usd",), "{:.4f}"),
)


def _get(scorecard: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = scorecard
    for key in path:
        value = value[key]
    return float(value)


def summarise(runs: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    """Per-metric values in run order with min, max and range. Refuses runs that differ
    in commit, dataset, prompts, models or threshold, or that repeat a run."""
    if len(runs) < 2:
        raise SystemExit("a noise band needs at least two runs")
    first = runs[0][1]
    for name, run in runs[1:]:
        for key in SAME:
            if run[key] != first[key]:
                raise SystemExit(f"{name} differs from {runs[0][0]} in {key}")
        if run["scorecard"]["git_sha"] != first["scorecard"]["git_sha"]:
            raise SystemExit(f"{name} differs from {runs[0][0]} in git_sha")
    created = [run["scorecard"]["created_at"] for _, run in runs]
    if len(set(created)) != len(created):
        raise SystemExit("the same run is listed twice")
    metrics = []
    for label, path, _fmt in METRICS:
        values = [_get(run["scorecard"], path) for _, run in runs]
        metrics.append(
            {
                "metric": label,
                "values": values,
                "min": min(values),
                "max": max(values),
                "range": max(values) - min(values),
            }
        )
    return {
        "git_sha": first["scorecard"]["git_sha"],
        "dataset": first["scorecard"]["dataset"],
        "n_examples": first["scorecard"]["n_examples"],
        "n_runs": len(runs),
        "runs": [name for name, _ in runs],
        "metrics": metrics,
    }


def render_block(summary: dict[str, Any]) -> str:
    fmt = {label: f for label, _path, f in METRICS}
    n_runs = summary["n_runs"]
    lines = [
        f"Commit `{summary['git_sha']}`, `{summary['dataset']}` split "
        f"(n={summary['n_examples']}), {n_runs} live runs with the same prompts, models "
        "and threshold. The first is the run of record; the others are reruns.",
        "",
        "| Metric | "
        + " | ".join(f"Run {i + 1}" for i in range(n_runs))
        + " | Min | Max | Range |",
        "|---|" + "---:|" * (n_runs + 3),
    ]
    for m in summary["metrics"]:
        f = fmt[m["metric"]]
        cells = [f.format(v) for v in m["values"]] + [
            f.format(m[k]) for k in ("min", "max", "range")
        ]
        lines.append(f"| {m['metric']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"n={n_runs} runs is a minimum, not a sample size: the range is a lower bound on the "
        "run-to-run spread, with no interval. A difference between two versions smaller than "
        "this range is not evidence of a change. Generated from "
        "[`datasets/eval/runs/noise/noise_band_6ec544d_dev.json`]"
        "(../datasets/eval/runs/noise/noise_band_6ec544d_dev.json) by "
        "`scripts/noise_band.py --check`; the reruns are pinned in "
        "`datasets/eval/runs/noise/SHA256SUMS`.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit 1 if a file would change")
    args = parser.parse_args(argv)
    runs = [
        (p.relative_to(REPO_ROOT).as_posix(), json.loads(p.read_text(encoding="utf-8")))
        for p in DEFAULT_RUNS
    ]
    summary = summarise(runs)
    summary_text = json.dumps(summary, indent=2) + "\n"
    doc = DOC.read_text(encoding="utf-8")
    rendered = replace_block(doc, render_block(summary), START, END)
    stale = [
        p
        for p, new, old in (
            (
                SUMMARY,
                summary_text,
                SUMMARY.read_text(encoding="utf-8") if SUMMARY.exists() else None,
            ),
            (DOC, rendered, doc),
        )
        if new != old
    ]
    if args.check:
        for path in stale:
            print(
                f"out of date: {path.relative_to(REPO_ROOT)} — run without --check", file=sys.stderr
            )
        return 1 if stale else 0
    SUMMARY.write_text(summary_text, encoding="utf-8")
    DOC.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
