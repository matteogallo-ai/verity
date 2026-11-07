"""Grade committed run artefacts for correctness against the reference answers.

    # paid: one judge call per answered answerable question, under a spending cap
    uv run python scripts/judge_correctness.py estimate --run <artefact> --dataset <jsonl>
    uv run python scripts/judge_correctness.py grade --run <artefact> --dataset <jsonl> --budget-usd 0.7
    # free: verdict files -> scorecards/correctness.json and the docs/results.md block
    uv run python scripts/judge_correctness.py render [--check]

Scope: the answerable questions an arm answered (refusals and unanswerable questions
have no reference to grade against). The agent is not called again: the answers are
the ones in the artefact. Verdict files go to ``datasets/eval/correctness/``, are
never overwritten, and record the judge model, the prompt hash, the artefact hash and
the spend. A run stopped by its cap is written as incomplete, with what it paid for.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from verity.eval.correctness import LLMCorrectnessJudge, numbers_match
from verity.eval.intervals import wilson
from verity.eval.run_store import load_run
from verity.types import EvalExample, PerExampleAudit

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "datasets" / "eval" / "correctness"
SCORECARD = REPO_ROOT / "scorecards" / "correctness.json"
DOC = REPO_ROOT / "docs" / "results.md"
START = "<!-- CORRECTNESS:START -->"
END = "<!-- CORRECTNESS:END -->"


def _rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _examples(dataset: Path) -> dict[str, EvalExample]:
    lines = dataset.read_text(encoding="utf-8").splitlines()
    return {e.id: e for e in (EvalExample.model_validate_json(x) for x in lines if x.strip())}


def to_grade(run: Path, dataset: Path) -> list[tuple[PerExampleAudit, EvalExample]]:
    """The answered answerable questions of ``run``, with their dataset example."""
    examples = _examples(dataset)
    pairs = []
    for audit in load_run(run).per_audit:
        if audit.refused or not audit.expected_answerable:
            continue
        example = examples.get(audit.example_id)
        if example is None or not example.reference_answer:
            raise SystemExit(f"{audit.example_id}: no reference answer in {_rel(dataset)}")
        pairs.append((audit, example))
    return pairs


def _out_path(run: Path) -> Path:
    return OUT_DIR / f"{run.stem}.correctness.json"


def estimate(run: Path, dataset: Path) -> int:
    from verity.eval.correctness import MAX_TOKENS
    from verity.llm import create_default_routing_client
    from verity.llm.budget import upper_cost_estimate
    from verity.llm.prompt import load_prompt
    from verity.llm.routing import create_judge_client

    client = create_judge_client(agent_client=create_default_routing_client())
    prompt = load_prompt("judge_correctness")
    pairs = to_grade(run, dataset)
    upper = sum(
        upper_cost_estimate(
            client.clients,
            prompt.render(
                question=e.question, reference_answer=e.reference_answer, answer=a.answer_text
            ),
            max_tokens=MAX_TOKENS,
        )
        for a, e in pairs
    )
    print(f"{len(pairs)} answers to grade; upper bound ${upper:.4f} (every call at max_tokens)")
    return 0


async def _grade(
    judge: LLMCorrectnessJudge, pairs: list[tuple[PerExampleAudit, EvalExample]]
) -> tuple[list[dict[str, Any]], str | None]:
    from verity.llm.budget import SpendCapExceededError

    verdicts: list[dict[str, Any]] = []
    for audit, example in pairs:
        try:
            v = await judge.grade(
                example.question, example.reference_answer or "", audit.answer_text
            )
        except SpendCapExceededError:
            return verdicts, "budget_exceeded"
        verdicts.append(
            {
                "example_id": audit.example_id,
                "verdict": v.verdict,
                "reason": v.reason,
                "cost_usd": round(v.cost_usd, 6),
            }
        )
    return verdicts, None


def grade(run: Path, dataset: Path, budget_usd: float) -> int:
    from verity.llm import create_default_routing_client
    from verity.llm.budget import SpendCappedClient
    from verity.llm.prompt import prompt_sha256
    from verity.llm.routing import create_judge_client, primary_client_identity

    out = _out_path(run)
    if out.exists():
        raise SystemExit(f"{_rel(out)} exists; verdict files are never overwritten")
    router = create_judge_client(agent_client=create_default_routing_client())
    capped = SpendCappedClient(router, cap_usd=budget_usd)
    pairs = to_grade(run, dataset)
    verdicts, incomplete = asyncio.run(_grade(LLMCorrectnessJudge(capped), pairs))
    payload = {
        "schema": "verity.correctness/v1",
        "run_artefact": _rel(run),
        "run_artefact_sha256": _sha256(run),
        "dataset": _rel(dataset),
        "dataset_sha256": _sha256(dataset),
        "judge_model": primary_client_identity(router)[1],
        "prompt_sha256": prompt_sha256()["judge_correctness"],
        "created_at": datetime.now(UTC).isoformat(),
        "budget_usd": budget_usd,
        "cost_usd": round(capped.spent.cost_usd, 6),
        "n_to_grade": len(pairs),
        "incomplete_reason": incomplete,
        "verdicts": verdicts,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {_rel(out)}: {len(verdicts)}/{len(pairs)} graded, ${capped.spent.cost_usd:.4f}")
    return 0


def _proportion(k: int, n: int) -> dict[str, Any]:
    ci = wilson(k, n)
    return {"k": k, "n": n, "ci95_wilson": None if ci is None else [round(x, 4) for x in ci]}


def number_checks(record: dict[str, Any]) -> dict[str, bool | None]:
    """The deterministic number check of each graded answer, recomputed from the
    artefact and the dataset rather than stored with the paid verdicts."""
    pairs = to_grade(REPO_ROOT / record["run_artefact"], REPO_ROOT / record["dataset"])
    return {a.example_id: numbers_match(e.reference_answer or "", a.answer_text) for a, e in pairs}


def summarise(record: dict[str, Any], numbers: dict[str, bool | None]) -> dict[str, Any]:
    vs = [{**v, "numbers_match": numbers[v["example_id"]]} for v in record["verdicts"]]
    n = len(vs)
    count = {
        label: sum(1 for v in vs if v["verdict"] == label)
        for label in ("correct", "partial", "incorrect", "unparsed")
    }
    with_numbers = [v for v in vs if v["numbers_match"] is not None]
    return {
        "run_artefact": record["run_artefact"],
        "judge_model": record["judge_model"],
        "cost_usd": record["cost_usd"],
        "incomplete_reason": record["incomplete_reason"],
        "n_to_grade": record["n_to_grade"],
        "correct": _proportion(count["correct"], n),
        "counts": count,
        "not_correct_ids": sorted(v["example_id"] for v in vs if v["verdict"] != "correct"),
        "numbers": {
            "n_with_numbers": len(with_numbers),
            "numbers_match": sum(1 for v in with_numbers if v["numbers_match"]),
            "disagree_ids": sorted(
                v["example_id"]
                for v in with_numbers
                if v["numbers_match"] != (v["verdict"] == "correct")
            ),
        },
    }


def _ids(ids: list[str]) -> str:
    return ", ".join(f"`{i}`" for i in ids) or "none"


def render_block(summaries: list[dict[str, Any]]) -> str:
    lines = [
        "| Run | Correct | Partial | Incorrect | Unparsed | Not correct | Numbers in the reference found in the answer |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        c, ci = s["correct"], s["correct"]["ci95_wilson"]
        interval = "" if ci is None else f" · [{ci[0]:.3f}, {ci[1]:.3f}]"
        nums = s["numbers"]
        graded = (
            c["n"]
            if s["incomplete_reason"] is None
            else (f"{c['n']} of {s['n_to_grade']}, {s['incomplete_reason']}")
        )
        lines.append(
            f"| `{s['run_artefact']}` ({graded}) | {c['k']}/{c['n']}{interval} | "
            f"{s['counts']['partial']} | {s['counts']['incorrect']} | {s['counts']['unparsed']} | "
            f"{_ids(s['not_correct_ids'])} | {nums['numbers_match']}/{nums['n_with_numbers']}; "
            f"disagrees with the grade on {_ids(nums['disagree_ids'])} |"
        )
    models = sorted({s["judge_model"] for s in summaries})
    cost = sum(s["cost_usd"] for s in summaries)
    lines += [
        "",
        f"Judge `{', '.join(models)}` with `prompts/judge_correctness.prompt`, total "
        f"${cost:.4f}. Scope: the answerable questions each run answered, graded against "
        "the dataset's `reference_answer`; the answers are the artefact's, the agent was "
        "not called again. The grade is **not validated against human labels**; the "
        "number check is deterministic but covers only references with a number. "
        "Generated from `scorecards/correctness.json` (`scripts/judge_correctness.py "
        "render --check`); verdicts and reasons in `datasets/eval/correctness/`.",
    ]
    return "\n".join(lines)


def _replace(text: str, block: str) -> str:
    if text.count(START) != 1 or text.count(END) != 1:
        raise SystemExit(f"{DOC.name} needs exactly one {START} and one {END}")
    head, rest = text.split(START)
    _, tail = rest.split(END)
    return f"{head}{START}\n{block}\n{END}{tail}"


def render(*, check: bool) -> int:
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUT_DIR.glob("*.json"))]
    for record in records:
        if _sha256(REPO_ROOT / record["run_artefact"]) != record["run_artefact_sha256"]:
            raise SystemExit(f"{record['run_artefact']} changed since it was graded")
    summaries = [summarise(r, number_checks(r)) for r in records]
    payload = json.dumps({"runs": summaries}, indent=2, sort_keys=True) + "\n"
    doc = DOC.read_text(encoding="utf-8")
    rendered = _replace(doc, render_block(summaries))
    current = SCORECARD.read_text(encoding="utf-8") if SCORECARD.is_file() else ""
    if check:
        stale = [
            n
            for n, ok in ((SCORECARD.name, current == payload), (DOC.name, doc == rendered))
            if not ok
        ]
        if stale:
            print(f"out of date: {', '.join(stale)} — run `render`", file=sys.stderr)
            return 1
        return 0
    SCORECARD.write_text(payload, encoding="utf-8")
    DOC.write_text(rendered, encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("estimate", "grade"):
        p = sub.add_parser(name)
        p.add_argument("--run", type=Path, required=True)
        p.add_argument("--dataset", type=Path, required=True)
        if name == "grade":
            p.add_argument("--budget-usd", type=float, required=True)
    r = sub.add_parser("render")
    r.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "estimate":
        return estimate(args.run, args.dataset)
    if args.command == "grade":
        return grade(args.run, args.dataset, args.budget_usd)
    return render(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
