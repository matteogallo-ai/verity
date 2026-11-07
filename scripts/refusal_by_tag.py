"""Refusal recall per question tag, with and without the easy unanswerable questions.

The aggregate refusal recall mixes questions a model declines without any check (a
football result or a drug dose asked of a company corpus: tag ``out-of-corpus``) with
those that test the design (``near-miss``, ``false-premise``), and with
``adversarial`` ones. This script splits the run of record by the
tags of its dataset, so each figure carries the questions it rests on.

It also publishes "unanswerable answered" under a second rule next to the rule of
record: an answer to an unanswerable question fails only when the judge finds one of
its claims unsupported (:func:`verity.eval.answer_quality.is_invented`). Under that
rule, an answer that states the premise is false, citing the passage, passes. The rule
of record stays: see ``docs/evaluation-methodology.md``.

    uv run python scripts/refusal_by_tag.py            # write the JSON and the docs block
    uv run python scripts/refusal_by_tag.py --check    # exit 1 if either is stale
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from verity.eval.answer_quality import is_invented
from verity.eval.intervals import wilson
from verity.eval.run_store import load_run
from verity.types import PerExampleAudit

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN = REPO_ROOT / "datasets" / "eval" / "runs" / "live" / "eval_969e1d5_test.json"
DATASET = REPO_ROOT / "datasets" / "eval" / "test.jsonl"
OUT = REPO_ROOT / "scorecards" / "refusal-by-tag.json"
DOC = REPO_ROOT / "docs" / "results.md"
START = "<!-- REFUSAL-BY-TAG:START -->"
END = "<!-- REFUSAL-BY-TAG:END -->"
# Unanswerable questions a model declines without any check, or built to attack the
# pipeline: the subsets below leave them out.
EASY_TAGS = ("out-of-corpus", "adversarial")


def _proportion(k: int, n: int) -> dict[str, Any]:
    ci = wilson(k, n)
    return {
        "k": k,
        "n": n,
        "ci95_wilson": None if ci is None else [round(ci[0], 4), round(ci[1], 4)],
    }


def analyse(audits: list[PerExampleAudit], tags: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    missing = sorted(a.example_id for a in audits if a.example_id not in tags)
    if missing:
        raise SystemExit(f"run questions not in the dataset: {', '.join(missing)}")
    unanswerable = [a for a in audits if not a.expected_answerable]
    answerable = [a for a in audits if a.expected_answerable]

    def recall(rows: list[PerExampleAudit]) -> dict[str, Any]:
        return {
            **_proportion(sum(a.refused for a in rows), len(rows)),
            "answered": sorted(a.example_id for a in rows if not a.refused),
        }

    by_tag: dict[str, Any] = {}
    for tag in sorted({t for a in unanswerable for t in tags[a.example_id]}):
        by_tag[tag] = recall([a for a in unanswerable if tag in tags[a.example_id]])
    false_refusals: dict[str, Any] = {}
    for tag in sorted({t for a in answerable for t in tags[a.example_id]}):
        rows = [a for a in answerable if tag in tags[a.example_id]]
        false_refusals[tag] = _proportion(sum(a.refused for a in rows), len(rows))

    def without(*excluded: str) -> list[PerExampleAudit]:
        return [a for a in unanswerable if not set(excluded) & set(tags[a.example_id])]

    answered = [a for a in unanswerable if not a.refused]
    return {
        "refusal_recall": {
            "all": recall(unanswerable),
            "without_out_of_corpus": recall(without("out-of-corpus")),
            "without_out_of_corpus_and_adversarial": recall(without(*EASY_TAGS)),
            "by_tag": by_tag,
        },
        "false_refusal_by_tag": false_refusals,
        "unanswerable_answered": {
            "rule_of_record": _proportion(len(answered), len(unanswerable)),
            "fails_only_if_invented": {
                **_proportion(sum(is_invented(a) for a in answered), len(unanswerable)),
                "invented": sorted(a.example_id for a in answered if is_invented(a)),
                "supported": sorted(a.example_id for a in answered if not is_invented(a)),
            },
        },
        "note": "a question can carry several tags; tag rows overlap and do not sum",
    }


def _cell(p: dict[str, Any]) -> str:
    ci = p["ci95_wilson"]
    interval = "" if ci is None else f" · [{ci[0]:.3f}, {ci[1]:.3f}]"
    return f"{p['k']}/{p['n']}{interval}"


def _ids(ids: list[str]) -> str:
    return ", ".join(f"`{i}`" for i in ids) or "none"


def render(report: dict[str, Any], run: str, dataset: str) -> str:
    rr = report["refusal_recall"]
    ua = report["unanswerable_answered"]
    alt = ua["fails_only_if_invented"]
    lines = [
        f"Run `{run}`, tags from `{dataset}`. Refusal recall = unanswerable questions "
        "refused / unanswerable questions, Wilson 95 % interval. A question can carry "
        "several tags, so tag rows overlap.",
        "",
        "| Unanswerable questions | Refused | Answered |",
        "|---|---|---|",
        f"| all | {_cell(rr['all'])} | {_ids(rr['all']['answered'])} |",
        f"| without `out-of-corpus` | {_cell(rr['without_out_of_corpus'])} | "
        f"{_ids(rr['without_out_of_corpus']['answered'])} |",
        f"| without `out-of-corpus` and `adversarial` | "
        f"{_cell(rr['without_out_of_corpus_and_adversarial'])} | "
        f"{_ids(rr['without_out_of_corpus_and_adversarial']['answered'])} |",
        *(
            f"| tag `{tag}` | {_cell(row)} | {_ids(row['answered'])} |"
            for tag, row in rr["by_tag"].items()
        ),
        "",
        "Answerable questions refused, by tag: "
        + " · ".join(f"`{t}` {_cell(p)}" for t, p in report["false_refusal_by_tag"].items())
        + ".",
        "",
        f"Unanswerable answered: {_cell(ua['rule_of_record'])} under the rule of record "
        "(any answer to an unanswerable question fails). If an answer failed only when "
        f"the judge finds one of its claims unsupported, it would be {_cell(alt)}: "
        f"{_ids(alt['invented'])} invented; {_ids(alt['supported'])} fully supported.",
        "",
        f"Generated from `{OUT.relative_to(REPO_ROOT).as_posix()}` "
        "(`scripts/refusal_by_tag.py --check`).",
    ]
    return "\n".join(lines)


def _tags(path: Path) -> dict[str, tuple[str, ...]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {r["id"]: tuple(r.get("tags", ())) for r in rows}


def _replace(text: str, block: str) -> str:
    if text.count(START) != 1 or text.count(END) != 1:
        raise SystemExit(f"{DOC.name} needs exactly one {START} and one {END}")
    head, rest = text.split(START)
    _, tail = rest.split(END)
    return f"{head}{START}\n{block}\n{END}{tail}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit 1 if an output is stale")
    args = parser.parse_args(argv)
    report = analyse(list(load_run(RUN).per_audit), _tags(DATASET))
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    rel = {p: p.relative_to(REPO_ROOT).as_posix() for p in (RUN, DATASET)}
    doc = DOC.read_text(encoding="utf-8")
    rendered = _replace(doc, render(report, rel[RUN], rel[DATASET]))
    current = OUT.read_text(encoding="utf-8") if OUT.is_file() else ""
    if args.check:
        stale = [
            n for n, ok in ((OUT.name, current == payload), (DOC.name, rendered == doc)) if not ok
        ]
        if stale:
            print(f"out of date: {', '.join(stale)} — run without --check", file=sys.stderr)
            return 1
        return 0
    OUT.write_text(payload, encoding="utf-8")
    DOC.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
