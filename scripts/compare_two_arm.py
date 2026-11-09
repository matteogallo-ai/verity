"""Verity against the naive RAG baseline on dev + test, as registered in
docs/preregistration/two-arm-comparison.md.

    # after the four runs: artefacts -> scorecard -> docs/results.md block
    uv run python scripts/compare_two_arm.py \
        --verity <dev artefact> <test artefact> --naive <dev artefact> <test artefact>
    # CI: scorecard == recomputed from the artefacts it names, block == rendered
    uv run python scripts/compare_two_arm.py --check

No model call. Each primary metric is a per-question outcome, paired between the arms
over the questions both completed, and tested with the exact McNemar test; the three
p-values are Holm-adjusted and read with the registered sentences.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from verity.eval.abstention import is_self_abstention
from verity.eval.answer_quality import is_invented
from verity.eval.headline import build_headline_dict_from_eval_run
from verity.eval.intervals import wilson
from verity.eval.paired import discordant, holm, mcnemar_exact
from verity.eval.run_store import load_run
from verity.types import PerExampleAudit

REPO_ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = "docs/preregistration/two-arm-comparison.md"
SCORECARD = REPO_ROOT / "scorecards" / "two-arm-comparison.json"
DOC = REPO_ROOT / "docs" / "results.md"
QUESTIONS = REPO_ROOT / "datasets" / "eval" / "questions.jsonl"
START, END = "<!-- TWO-ARM:START -->", "<!-- TWO-ARM:END -->"
ALPHA = 0.05
ARMS = ("verity", "naive")


def _rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def outcome(audit: PerExampleAudit) -> dict[str, Any]:
    """The registered per-question outcomes. A self-abstention counts as a refusal; an
    answer the judge did not score has no invented outcome and leaves metric 3."""
    shipped = not audit.refused
    judged = audit.judge_claims is not None
    answered = shipped and not is_self_abstention(audit.answer_text)
    invented: bool | None
    if not answered:
        invented = False
    elif audit.judge_claims is None:
        invented = None
    else:
        invented = is_invented(audit)
    return {
        "answerable": audit.expected_answerable,
        "shipped": shipped,
        "answered": answered,
        "invented": invented,
        "judged_shipped": shipped and judged,
        "invented_shipped": (shipped and is_invented(audit)) if judged or not shipped else None,
    }


def _ratio(k: int, n: int) -> dict[str, Any]:
    ci = wilson(k, n)
    return {"k": k, "n": n, "ci95_wilson": None if ci is None else [round(x, 4) for x in ci]}


# (key, label, scope, outcome field, lower is better)
PRIMARY = (
    ("unanswerable_answered", "Unanswerable answered", "unanswerable", "answered", True),
    ("answerable_answered", "Answerable answered", "answerable", "answered", False),
    (
        "invented_answer",
        "Invented answer, self-abstentions as refusals",
        "all",
        "invented",
        True,
    ),
)
SECONDARY = (
    (
        "unanswerable_shipped",
        "Unanswerable answered, self-abstentions as answers",
        "unanswerable",
        "shipped",
    ),
    (
        "answerable_shipped",
        "Answerable answered, self-abstentions as answers",
        "answerable",
        "shipped",
    ),
    ("invented_shipped", "Invented answer, self-abstentions as answers", "all", "invented_shipped"),
)


def _in_scope(o: dict[str, Any], scope: str) -> bool:
    return scope == "all" or o["answerable"] == (scope == "answerable")


def _row(
    per_arm: dict[str, dict[str, dict[str, Any]]], shared: list[str], scope: str, field: str
) -> dict[str, Any]:
    ids = [
        q
        for q in shared
        if _in_scope(per_arm["verity"][q], scope)
        and all(per_arm[a][q][field] is not None for a in ARMS)
    ]
    flags = {a: {q: bool(per_arm[a][q][field]) for q in ids} for a in ARMS}
    v_only, n_only, n = discordant(flags["verity"], flags["naive"])
    return {
        **{a: _ratio(sum(flags[a].values()), n) for a in ARMS},
        "verity_only": v_only,
        "naive_only": n_only,
        "n_paired": n,
        "p_mcnemar": mcnemar_exact(v_only, n_only),
        "verity_only_ids": sorted(q for q in ids if flags["verity"][q] and not flags["naive"][q]),
        "naive_only_ids": sorted(q for q in ids if flags["naive"][q] and not flags["verity"][q]),
    }


def reading(key: str, row: dict[str, Any]) -> str:
    """The sentences of the registered decision rule."""
    significant = row["p_holm"] < ALPHA
    verity_lower = row["verity"]["k"] < row["naive"]["k"]
    n = row["n_paired"]
    if key == "unanswerable_answered":
        if not significant:
            return (
                f"On {n} unanswerable questions the checks show no effect beyond the "
                "synthesizer's own abstentions."
            )
        if verity_lower:
            return "Verity's checks refuse unanswerable questions that the naive draft answers."
        return "Verity answers more unanswerable questions than the naive draft: the checks are a defect here."
    if key == "answerable_answered":
        if not significant:
            return "No cost measurable at this n."
        if verity_lower:
            return "The checks cost answerable answers."
        return "Verity answers more answerable questions than the naive draft."
    if not significant:
        return "No difference in invented answers measurable at this n."
    return "Verity invents fewer answers." if verity_lower else "Verity invents more answers."


def _arm_summary(paths: list[Path]) -> dict[str, Any]:
    runs = [load_run(p) for p in paths]
    heads = [
        build_headline_dict_from_eval_run(r, source_artefact=_rel(p))
        for r, p in zip(runs, paths, strict=True)
    ]
    n = sum(len(r.per_audit) for r in runs)
    agent = sum(h["cost"]["agent_usd"] or 0.0 for h in heads)
    judge = sum(h["cost"]["judge_usd"] or 0.0 for h in heads)
    return {
        "artefacts": [
            {
                "path": _rel(p),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "git_sha": r.scorecard.git_sha,
                "n_completed": len(r.per_audit),
                "incomplete_reason": r.scorecard.incomplete_reason,
            }
            for r, p in zip(runs, paths, strict=True)
        ],
        "agent_model": sorted({r.agent_model for r in runs}),
        "judge_model": sorted({r.judge_model for r in runs}),
        "cost": {
            "agent_usd": round(agent, 6),
            "judge_usd": round(judge, 6),
            "per_query_agent_usd": round(agent / n, 6) if n else None,
            "per_query_with_judge_usd": round((agent + judge) / n, 6) if n else None,
        },
    }


def analyse(verity: list[Path], naive: list[Path]) -> dict[str, Any]:
    audits = {
        "verity": [a for p in verity for a in load_run(p).per_audit],
        "naive": [a for p in naive for a in load_run(p).per_audit],
    }
    per_arm = {arm: {a.example_id: outcome(a) for a in audits[arm]} for arm in ARMS}
    shared = sorted(set(per_arm["verity"]) & set(per_arm["naive"]))
    only_one = sorted(set(per_arm["verity"]) ^ set(per_arm["naive"]))
    primary = {key: _row(per_arm, shared, scope, field) for key, _, scope, field, _ in PRIMARY}
    for key, adjusted in zip(
        primary, holm([r["p_mcnemar"] for r in primary.values()]), strict=True
    ):
        primary[key]["p_holm"] = adjusted
        primary[key]["reading"] = reading(key, primary[key])
    secondary = {key: _row(per_arm, shared, scope, field) for key, _, scope, field in SECONDARY}
    # Rounded only for storage, after Holm has used the exact values.
    for row in (*primary.values(), *secondary.values()):
        for p_key in ("p_mcnemar", "p_holm"):
            if p_key in row:
                row[p_key] = round(row[p_key], 10)
    tags = {
        r["id"]: tuple(r.get("tags", ()))
        for r in (json.loads(x) for x in QUESTIONS.read_text(encoding="utf-8").splitlines() if x)
    }
    unanswerable = [q for q in shared if not per_arm["verity"][q]["answerable"]]
    by_tag = {
        tag: {
            arm: _ratio(sum(1 for q in ids if per_arm[arm][q]["answered"]), len(ids))
            for arm in ARMS
        }
        for tag in sorted({t for q in unanswerable for t in tags[q]})
        if (ids := [q for q in unanswerable if tag in tags[q]])
    }
    hallucination = {
        arm: _ratio(
            sum(
                1
                for q in shared
                if per_arm[arm][q]["answerable"] and per_arm[arm][q]["invented_shipped"]
            ),
            sum(
                1
                for q in shared
                if per_arm[arm][q]["answerable"] and per_arm[arm][q]["judged_shipped"]
            ),
        )
        for arm in ARMS
    }
    shipped_rate = {
        arm: _ratio(
            sum(1 for q in shared if per_arm[arm][q]["invented_shipped"]),
            sum(1 for q in shared if per_arm[arm][q]["judged_shipped"]),
        )
        for arm in ARMS
    }
    return {
        "preregistration": PREREGISTRATION,
        "arms": {"verity": _arm_summary(verity), "naive": _arm_summary(naive)},
        "n_compared": len(shared),
        "completed_by_one_arm_only": only_one,
        "primary": primary,
        "secondary": secondary,
        "invented_over_shipped_answers": shipped_rate,
        "hallucination_rate_answerable": hallucination,
        "unanswerable_answered_by_tag": by_tag,
    }


def fmt_p(p: float) -> str:
    """Four decimals; a p-value below that precision is not shown as zero."""
    return "< 0.0001" if p < 0.00005 else f"{p:.4f}"


def _cell(r: dict[str, Any]) -> str:
    ci = r["ci95_wilson"]
    return f"{r['k']}/{r['n']}" + ("" if ci is None else f" [{ci[0]:.3f}, {ci[1]:.3f}]")


def render(report: dict[str, Any]) -> str:
    v, nv = report["arms"]["verity"], report["arms"]["naive"]
    shas = sorted({a["git_sha"] for arm in (v, nv) for a in arm["artefacts"]})
    lines = [
        f"{report['n_compared']} questions completed by both arms (dev + test), agent "
        f"`{', '.join(v['agent_model'])}`, judge `{', '.join(v['judge_model'])}`, commit "
        f"`{', '.join(shas)}`. [Pre-registered]({PREREGISTRATION.removeprefix('docs/')}) before "
        "the runs: a self-abstention counts as a refusal; exact McNemar on the paired "
        "questions, Holm-adjusted over the three primary tests.",
        "",
        "| Metric | Better | Verity | Naive RAG | Verity only / naive only | p (Holm) | Reading |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, label, _, _, lower in PRIMARY:
        r = report["primary"][key]
        lines.append(
            f"| **{label}** | {'lower' if lower else 'higher'} | {_cell(r['verity'])} | "
            f"{_cell(r['naive'])} | {r['verity_only']} / {r['naive_only']} | "
            f"{fmt_p(r['p_mcnemar'])} ({fmt_p(r['p_holm'])}) | {r['reading']} |"
        )
    for key, label, _, _ in SECONDARY:
        r = report["secondary"][key]
        lines.append(
            f"| {label} | — | {_cell(r['verity'])} | {_cell(r['naive'])} | "
            f"{r['verity_only']} / {r['naive_only']} | {fmt_p(r['p_mcnemar'])} | descriptive |"
        )
    s = report["invented_over_shipped_answers"]
    lines.append(
        f"| Invented, over the answers each arm shipped | — | {_cell(s['verity'])} | "
        f"{_cell(s['naive'])} | — | — | not paired |"
    )
    h = report["hallucination_rate_answerable"]
    lines.append(
        f"| Hallucination rate, answerable questions answered | — | {_cell(h['verity'])} | "
        f"{_cell(h['naive'])} | — | — | not paired |"
    )
    lines += [
        "",
        "Unanswerable answered by tag (self-abstentions as refusals; a question can carry "
        "several tags), descriptive:",
        "",
        "| Tag | Verity | Naive RAG |",
        "|---|---|---|",
    ]
    for tag, row in report["unanswerable_answered_by_tag"].items():
        lines.append(f"| `{tag}` | {_cell(row['verity'])} | {_cell(row['naive'])} |")
    lines += [""]
    for arm, label in (("verity", "Verity"), ("naive", "Naive RAG")):
        a = report["arms"][arm]
        c = a["cost"]
        incomplete = [
            f"`{Path(x['path']).name}` stopped after {x['n_completed']} ({x['incomplete_reason']})"
            for x in a["artefacts"]
            if x["incomplete_reason"]
        ]
        lines.append(
            f"- {label}: ${c['agent_usd']:.4f} agent, ${c['judge_usd']:.4f} judge; per query "
            f"${c['per_query_agent_usd']:.4f} agent only, ${c['per_query_with_judge_usd']:.4f} "
            "with the judge." + (" " + "; ".join(incomplete) + "." if incomplete else "")
        )
    if report["completed_by_one_arm_only"]:
        ids = ", ".join(f"`{q}`" for q in report["completed_by_one_arm_only"])
        lines.append(f"- Completed by one arm only, left out of every row: {ids}.")
    p1 = report["primary"]["unanswerable_answered"]
    lines.append(
        "- Discordant questions, unanswerable answered: Verity only "
        + (", ".join(f"`{q}`" for q in p1["verity_only_ids"]) or "none")
        + "; naive only "
        + (", ".join(f"`{q}`" for q in p1["naive_only_ids"]) or "none")
        + "."
    )
    lines += [
        "",
        f"Generated from `{_rel(SCORECARD)}`, computed from the four artefacts it names "
        "(`scripts/compare_two_arm.py --check`).",
    ]
    return "\n".join(lines)


def _replace(text: str, block: str) -> str:
    if text.count(START) != 1 or text.count(END) != 1:
        raise SystemExit(f"{DOC.name} needs exactly one {START} and one {END}")
    head, rest = text.split(START)
    _, tail = rest.split(END)
    return f"{head}{START}\n{block}\n{END}{tail}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--verity", type=Path, nargs="+")
    parser.add_argument("--naive", type=Path, nargs="+")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        committed = json.loads(SCORECARD.read_text(encoding="utf-8"))
        verity, naive = (
            [REPO_ROOT / x["path"] for x in committed["arms"][arm]["artefacts"]] for arm in ARMS
        )
    elif args.verity and args.naive:
        verity, naive = args.verity, args.naive
    else:
        parser.error("--verity and --naive, or --check")
    report = analyse(verity, naive)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    doc = DOC.read_text(encoding="utf-8")
    rendered = _replace(doc, render(report))
    if args.check:
        stale = [
            n
            for n, ok in (
                (SCORECARD.name, SCORECARD.read_text(encoding="utf-8") == payload),
                (DOC.name, doc == rendered),
            )
            if not ok
        ]
        if stale:
            print(f"out of date: {', '.join(stale)}", file=sys.stderr)
            return 1
        return 0
    SCORECARD.write_text(payload, encoding="utf-8")
    DOC.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
