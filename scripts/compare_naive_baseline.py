"""Compare Verity with the naive RAG baseline under the pre-registered rule.

    # after both runs: artefacts -> comparison scorecard -> README block
    uv run python scripts/compare_naive_baseline.py \
        --verity datasets/eval/runs/live/eval_<sha>_dev.json \
        --naive datasets/eval/runs/live/eval_<sha>_dev-naive-baseline.json
    # CI: comparison scorecard == recomputed from its artefacts, README block == rendered
    uv run python scripts/compare_naive_baseline.py --check

Metrics, abstention patterns and the reading rule are those of
``docs/preregistration/naive-rag-baseline.md``: each metric is k/n with a Wilson 95 %
interval per arm; overlapping intervals read "not distinguishable", disjoint ones
"distinguishable". The per-arm figures come from the same scorecard builder as the
published results, so a metric means the same thing here as in ``docs/results.md``.
The README block sits between ``<!-- BASELINE:START -->`` and ``<!-- BASELINE:END -->``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from verity.eval.headline import build_headline_dict_from_eval_run
from verity.eval.intervals import wilson
from verity.eval.run_store import load_run
from verity.types import EvalRun

REPO_ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = "docs/preregistration/naive-rag-baseline.md"
DEFAULT_SCORECARD = REPO_ROOT / "scorecards" / "naive-baseline-comparison.json"
DEFAULT_README = REPO_ROOT / "README.md"
START, END = "<!-- BASELINE:START -->", "<!-- BASELINE:END -->"
# Copied verbatim from the pre-registration; a test holds the two equal.
ABSTENTION_PATTERNS = (
    r"\b(do|does|did) not (contain|mention|provide|include|specify|say|state|address|cover|describe)\b",
    r"\b(don't|doesn't) (contain|mention|provide|include|specify|say|state|address|cover|describe)\b",
    r"\bno (information|mention|details?|data)\b",
    r"\b(is|are|was|were) not (mentioned|specified|stated|provided|covered|available|included|described)\b",
    r"\b(cannot|can't|unable to) (be )?(determine|answer|find|confirm)",
    r"\bI don't know\b",
)
_ABSTENTION_RE = [re.compile(p, re.IGNORECASE) for p in ABSTENTION_PATTERNS]


def is_self_abstention(text: str) -> bool:
    return any(p.search(text) for p in _ABSTENTION_RE)


def _ratio(k: int, n: int) -> dict[str, Any]:
    ci = wilson(k, n)
    return {
        "k": k,
        "n": n,
        "value": None if n == 0 else round(k / n, 6),
        "ci95_wilson": None if ci is None else [round(ci[0], 6), round(ci[1], 6)],
    }


def arm_metrics(run: EvalRun, path: Path) -> dict[str, Any]:
    """The pre-registered metrics of one arm, from its artefact alone."""
    source = _repo_path(path)
    # The pre-registration compares an incomplete arm on the questions both arms
    # completed; that path is not built because no run has needed it: fail, never
    # compare two different sets of questions in silence.
    if run.scorecard.errored_example_ids:
        raise SystemExit(f"{source}: errored examples {run.scorecard.errored_example_ids}")
    if run.scorecard.incomplete_reason is not None:
        raise SystemExit(f"{source}: incomplete run ({run.scorecard.incomplete_reason})")
    head = build_headline_dict_from_eval_run(run, source_artefact=source)
    refu, quality = head["refusal"], head["answer_quality"]
    unanswerable = [a for a in run.per_audit if not a.expected_answerable]
    answerable = [a for a in run.per_audit if a.expected_answerable]
    ooa = refu["out_of_scope_answered"]
    hall = quality["hallucination_rate"]
    self_abstained = [
        a for a in unanswerable if not a.refused and is_self_abstention(a.answer_text)
    ]
    answered_unanswerable = sum(1 for a in unanswerable if not a.refused)
    return {
        "source_artefact": source,
        "artefact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "git_sha": run.scorecard.git_sha,
        "agent_model": run.agent_model,
        "judge_model": run.judge_model,
        "n_examples": len(run.per_audit),
        "total_cost_usd": round(head["cost"]["total_usd"] or 0.0, 6),
        "metrics": {
            "unanswerable_answered": _ratio(ooa["count"], ooa["total"]),
            "hallucination_rate": _ratio(hall["k"], hall["n_questions"]),
            "answerable_answered": _ratio(
                head["sample_sizes"]["n_answerable_answered"], len(answerable)
            ),
            "unanswerable_answered_without_self_abstentions": _ratio(
                answered_unanswerable - len(self_abstained), len(unanswerable)
            ),
            "faithfulness_per_claim": _ratio(
                quality["faithfulness"]["micro"]["k"], quality["faithfulness"]["micro"]["n_claims"]
            ),
            "citation_accuracy_per_claim": _ratio(
                quality["citation_accuracy"]["micro"]["k"],
                quality["citation_accuracy"]["micro"]["n_claims"],
            ),
        },
        "self_abstained_ids": sorted(a.example_id for a in self_abstained),
    }


# (key, label, better direction, primary?)
ROWS = (
    ("unanswerable_answered", "Unanswerable answered", "lower", True),
    ("hallucination_rate", "Hallucination rate (answered answerable)", "lower", True),
    ("answerable_answered", "Answerable answered", "higher", True),
    (
        "unanswerable_answered_without_self_abstentions",
        "Unanswerable answered, self-abstentions as refusals",
        "lower",
        False,
    ),
    ("faithfulness_per_claim", "Faithfulness, per claim", "higher", False),
    ("citation_accuracy_per_claim", "Citation accuracy, per claim", "higher", False),
)


def read(verity: dict[str, Any], naive: dict[str, Any]) -> str:
    """The pre-registered reading: Wilson 95 % intervals overlap or they do not."""
    a, b = verity["ci95_wilson"], naive["ci95_wilson"]
    if a is None or b is None:
        return "not computable (n = 0)"
    if a[0] <= b[1] and b[0] <= a[1]:
        return "not distinguishable"
    direction = "higher" if verity["value"] > naive["value"] else "lower"
    return f"distinguishable: Verity {direction}"


def build(verity_path: Path, naive_path: Path) -> dict[str, Any]:
    arms = {
        "verity": arm_metrics(load_run(verity_path), verity_path),
        "naive": arm_metrics(load_run(naive_path), naive_path),
    }
    readings = {
        key: read(arms["verity"]["metrics"][key], arms["naive"]["metrics"][key]) for key, *_ in ROWS
    }
    return {
        "schema": "verity.naive-baseline-comparison/v1",
        "preregistration": PREREGISTRATION,
        "arms": arms,
        "readings": readings,
    }


def _repo_path(path: Path) -> str:
    """``path`` relative to the repository when inside it, else absolute."""
    resolved = path.resolve()
    inside = resolved.is_relative_to(REPO_ROOT)
    return (resolved.relative_to(REPO_ROOT) if inside else resolved).as_posix()


def _cell(m: dict[str, Any]) -> str:
    if m["ci95_wilson"] is None:
        return f"{m['k']}/{m['n']}"
    lo, hi = m["ci95_wilson"]
    return f"{m['k']}/{m['n']} = {m['value']:.3f} [{lo:.3f}, {hi:.3f}]"


def render_block(data: dict[str, Any], scorecard_path: Path = DEFAULT_SCORECARD) -> str:
    v, n = data["arms"]["verity"], data["arms"]["naive"]
    lines = [
        f"Dev split, n = {v['n_examples']}, same questions, agent `{v['agent_model']}`, "
        f"judge `{v['judge_model']}`, both arms at commit `{v['git_sha']}`. The naive arm "
        "ships the synthesizer's draft with no citation gate, support check, scorer or "
        f"threshold. Metrics and reading rule were [pre-registered]({PREREGISTRATION}) "
        "before the runs: Wilson 95 % intervals that overlap read *not distinguishable*.",
        "",
        *(
            ["> ⚠️ **Generated on stubs**: these figures describe the scripted stubs.", ""]
            if "stub" in v["agent_model"] or "stub" in v["judge_model"]
            else []
        ),
        "| Metric | Better | Verity | Naive RAG | Reading |",
        "|---|---|---|---|---|",
    ]
    for key, label, better, primary in ROWS:
        name = f"**{label}**" if primary else label
        lines.append(
            f"| {name} | {better} | {_cell(v['metrics'][key])} | {_cell(n['metrics'][key])} "
            f"| {data['readings'][key]} |"
        )
    lines += [
        "",
        "Bold rows are the primary metrics. Per-claim intervals treat the claims of one "
        "answer as independent, so they are too narrow. Cost: Verity "
        f"${v['total_cost_usd']:.4f}, naive ${n['total_cost_usd']:.4f}, agent and judge.",
        "",
        f"Generated from [`{_repo_path(scorecard_path)}`]({_repo_path(scorecard_path)}), "
        f"computed from `{v['source_artefact']}` and `{n['source_artefact']}` "
        "(`scripts/compare_naive_baseline.py --check`).",
    ]
    return "\n".join(lines)


def replace_block(text: str, block: str) -> str:
    i, j = text.find(START), text.find(END)
    if i == -1 or j == -1 or j < i:
        raise SystemExit(f"README missing {START!r} / {END!r} markers")
    return f"{text[: i + len(START)]}\n{block}\n{text[j:]}"


def _dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verity", type=Path)
    parser.add_argument("--naive", type=Path)
    parser.add_argument("--scorecard", type=Path, default=DEFAULT_SCORECARD)
    parser.add_argument("--readme", type=Path, default=DEFAULT_README)
    parser.add_argument("--check", action="store_true", help="exit 1 if a file would change")
    args = parser.parse_args(argv)
    existing = json.loads(args.scorecard.read_text()) if args.scorecard.is_file() else None
    if args.verity is None or args.naive is None:
        if existing is None:
            raise SystemExit(f"no --verity/--naive given and {args.scorecard} does not exist")
        verity = REPO_ROOT / existing["arms"]["verity"]["source_artefact"]
        naive = REPO_ROOT / existing["arms"]["naive"]["source_artefact"]
    else:
        verity, naive = args.verity, args.naive
    payload = build(verity, naive)
    readme = args.readme.read_text(encoding="utf-8")
    rendered = replace_block(readme, render_block(payload, args.scorecard))
    pending = []
    if existing is None or _dump(existing) != _dump(payload):
        pending.append((args.scorecard, _dump(payload)))
    if rendered != readme:
        pending.append((args.readme, rendered))
    if args.check:
        if pending:
            names = ", ".join(str(p) for p, _ in pending)
            print(f"out of date: {names} — run without --check", file=sys.stderr)
            return 1
        print(f"comparison and README match {verity.name} and {naive.name}")
        return 0
    for path, text in pending:
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
