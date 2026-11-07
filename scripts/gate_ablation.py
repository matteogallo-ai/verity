"""Each deterministic refusal check, one at a time: how many right and wrong refusals it makes.

No model call. Two readings of committed artefacts:

1. **First failing check, on Verity's runs.** A refused record names the first check
   that failed (``refusal_reason``) but keeps neither the draft nor its citations, so a
   later check cannot be replayed on it. This counts each cause against the gold label:
   a refusal of an unanswerable question is right, of an answerable one wrong.
2. **Every check on its own, on the naive arm's drafts.** The naive baseline arm runs
   the same pipeline and ships its drafts unchecked, with their citations and the
   reranker scores of the hits (``hits_used``). Every check is replayed on every draft
   with the thresholds of ``verity.config``: alone, and with all checks but that one.

The replay is checked against the agent before it is trusted: on Verity's answered
records every check must pass, since the agent shipped them. A mismatch exits 1.

    uv run python scripts/gate_ablation.py            # write the JSON and the docs block
    uv run python scripts/gate_ablation.py --check    # exit 1 if either is stale
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from verity.agent.support import (
    best_cited_rerank_score,
    citation_coverage,
    suspected_documents,
)
from verity.config import Settings
from verity.eval.run_store import load_run
from verity.types import PerExampleAudit

REPO_ROOT = Path(__file__).resolve().parents[1]
LIVE = REPO_ROOT / "datasets" / "eval" / "runs" / "live"
NOISE = REPO_ROOT / "datasets" / "eval" / "runs" / "noise"
VERITY_RUNS = {
    "dev": [
        LIVE / "eval_6ec544d_dev.json",
        NOISE / "eval_6ec544d_dev.rerun2.json",
        NOISE / "eval_6ec544d_dev.rerun3.json",
        LIVE / "eval_e644a98_dev.incomplete-20261105T094431636600Z.json",
    ],
    "test": [LIVE / "eval_969e1d5_test.json"],
}
NAIVE_RUNS = {
    "dev": [LIVE / "eval_e644a98_dev-naive-baseline.incomplete-20261105T094823388125Z.json"],
}
OUT = REPO_ROOT / "scorecards" / "gate-ablation.json"
DOC = REPO_ROOT / "docs" / "results.md"
START = "<!-- GATE-ABLATION:START -->"
END = "<!-- GATE-ABLATION:END -->"
# Pipeline order of verity.agent.agent and verity.agent.support.check_support.
GATES = ("no_citation", "suspected_injection", "low_citation_coverage", "irrelevant_citation")


def _rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def gate_fails(audit: PerExampleAudit, settings: Settings) -> dict[str, bool]:
    """Whether each check would refuse this draft, computed as the agent computes it."""
    hits = list(audit.hits_used)
    citations = audit.citations
    suspected = suspected_documents(hits)
    rerank = best_cited_rerank_score(citations, hits)
    return {
        "no_citation": not citations or not audit.answer_text.strip(),
        "suspected_injection": any(str(c.document_id) in suspected for c in citations),
        "low_citation_coverage": citation_coverage(audit.answer_text, citations)
        < settings.min_citation_coverage,
        "irrelevant_citation": rerank is not None and rerank < settings.min_cited_rerank_score,
    }


def _tally(rows: list[tuple[bool, bool]]) -> dict[str, int]:
    """``rows`` of ``(refused, answerable)``: right and wrong refusals."""
    return {
        "right": sum(1 for refused, answerable in rows if refused and not answerable),
        "wrong": sum(1 for refused, answerable in rows if refused and answerable),
    }


def first_failing(paths: list[Path]) -> dict[str, Any]:
    """Refusals per first failing check, as records and as distinct questions: the dev
    runs include reruns of the same questions, so records overstate the evidence."""
    rows: dict[str, dict[str, Any]] = {}
    n = 0
    for path in paths:
        for a in load_run(path).per_audit:
            n += 1
            if not a.refused:
                continue
            row = rows.setdefault(
                str(a.refusal_reason),
                {"right": 0, "wrong": 0, "right_ids": set(), "wrong_ids": set()},
            )
            side = "wrong" if a.expected_answerable else "right"
            row[side] += 1
            row[f"{side}_ids"].add(a.example_id)
    by_cause = {
        cause: {**r, "right_ids": sorted(r["right_ids"]), "wrong_ids": sorted(r["wrong_ids"])}
        for cause, r in sorted(rows.items())
    }
    right = sum(r["right"] for r in by_cause.values())
    refused = sum(r["right"] + r["wrong"] for r in by_cause.values())
    dropped = by_cause.get("irrelevant_citation", {"right": 0, "wrong": 0})
    return {
        "runs": [_rel(p) for p in paths],
        "n_records": n,
        "by_cause": by_cause,
        # With the check off, each record it refused becomes an answer or a scorer
        # refusal. Precision is lowest when its right refusals become answers and its
        # wrong ones stay refused, highest the other way round.
        "without_irrelevant_citation": {
            "lowest": [right - dropped["right"], refused - dropped["right"]],
            "highest": [right, refused - dropped["wrong"]],
        },
    }


def check_replay(paths: list[Path], settings: Settings) -> list[str]:
    """Answered records on which the replay says a check fails: the agent shipped them,
    so each one is a disagreement between this script and the agent."""
    wrong = []
    for path in paths:
        run = load_run(path)
        for a in run.per_audit:
            if not a.refused and any(gate_fails(a, settings).values()):
                wrong.append(f"{path.name}:{a.example_id}")
    return wrong


def replay(paths: list[Path], settings: Settings) -> dict[str, Any]:
    drafts = [a for p in paths for a in load_run(p).per_audit]
    fails = [(gate_fails(a, settings), a.expected_answerable, a.example_id) for a in drafts]

    def counts(refuses: Callable[[dict[str, bool]], bool]) -> dict[str, Any]:
        refused = [(f, ans, qid) for f, ans, qid in fails if refuses(f)]
        return {
            **_tally([(True, ans) for _, ans, _ in refused]),
            "wrong_ids": sorted(qid for _, ans, qid in refused if ans),
        }

    return {
        "runs": [_rel(p) for p in paths],
        "n_drafts": len(drafts),
        "n_answerable": sum(1 for a in drafts if a.expected_answerable),
        "thresholds": {
            "min_citation_coverage": settings.min_citation_coverage,
            "min_cited_rerank_score": settings.min_cited_rerank_score,
        },
        "alone": {g: counts(lambda f, g=g: f[g]) for g in GATES},
        "all": counts(lambda f: any(f.values())),
        "all_but": {
            g: counts(lambda f, g=g: any(v for k, v in f.items() if k != g)) for g in GATES
        },
    }


def analyse(settings: Settings) -> dict[str, Any]:
    mismatches = check_replay([p for ps in VERITY_RUNS.values() for p in ps], settings)
    if mismatches:
        raise SystemExit(f"replay disagrees with the agent on: {', '.join(mismatches)}")
    return {
        "first_failing": {split: first_failing(ps) for split, ps in VERITY_RUNS.items()},
        "replay_on_naive_drafts": {split: replay(ps, settings) for split, ps in NAIVE_RUNS.items()},
        "replay_checked_on_answered_records": True,
    }


def _rw(row: dict[str, Any]) -> str:
    return f"{row['right']} / {row['wrong']}"


def _rw_questions(row: dict[str, Any]) -> str:
    return f"{_rw(row)} ({len(row['right_ids'])} / {len(row['wrong_ids'])})"


def default_settings() -> Settings:
    """The thresholds the runs used: the code defaults, read neither from .env nor from
    the caller's VERITY_ variables, which would silently change every replayed count."""
    return Settings.model_construct()


def render(report: dict[str, Any]) -> str:
    lines = [
        "Right / wrong refusals: a refusal of an unanswerable question is right, of an "
        "answerable one wrong.",
        "",
        "**First failing check, on Verity's runs.** A refused record keeps only the first "
        "check that failed, so a later check is never seen on it. Each cell is records, "
        "then distinct questions in brackets: the dev runs include reruns of the same "
        "questions.",
        "",
        "| Check | "
        + " | ".join(f"{s} ({r['n_records']} records)" for s, r in report["first_failing"].items())
        + " |",
        "|---|" + "---|" * len(report["first_failing"]),
    ]
    causes = sorted({c for r in report["first_failing"].values() for c in r["by_cause"]})
    for cause in causes:
        cells = [
            _rw_questions(
                r["by_cause"].get(cause, {"right": 0, "wrong": 0, "right_ids": [], "wrong_ids": []})
            )
            for r in report["first_failing"].values()
        ]
        lines.append(f"| `{cause}` | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "Runs: "
        + "; ".join(
            f"{s}: " + ", ".join(f"`{Path(p).name}`" for p in r["runs"])
            for s, r in report["first_failing"].items()
        )
        + "."
    )
    for split, r in report["first_failing"].items():
        lo, hi = (r["without_irrelevant_citation"][k] for k in ("lowest", "highest"))
        lines += [
            "",
            f"With `irrelevant_citation` off, refusal precision on {split} lies between "
            f"{lo[0]}/{lo[1]} and {hi[0]}/{hi[1]} (records): each record it refused "
            "becomes an answer or a scorer refusal.",
        ]
    for split, r in report["replay_on_naive_drafts"].items():
        th = r["thresholds"]
        lines += [
            "",
            f"**Every check on its own, on the naive arm's {split} drafts** ({r['n_drafts']} "
            f"drafts, {r['n_answerable']} answerable; thresholds: coverage "
            f"{th['min_citation_coverage']}, cited reranker score {th['min_cited_rerank_score']}). "
            "The naive arm ran the same pipeline and shipped its drafts unchecked, so each "
            "check can be replayed on every draft. A draft with no citation has coverage "
            "0, so `low_citation_coverage` alone also counts the `no_citation` drafts.",
            "",
            "| Check | Alone: right / wrong | All checks but this one: right / wrong | Wrongly refused alone |",
            "|---|---|---|---|",
        ]
        for g in GATES:
            alone, without = r["alone"][g], r["all_but"][g]
            ids = ", ".join(f"`{i}`" for i in alone["wrong_ids"]) or "none"
            lines.append(f"| `{g}` | {_rw(alone)} | {_rw(without)} | {ids} |")
        lines.append(f"| all four | {_rw(r['all'])} | — | — |")
    lines += [
        "",
        "The replay is checked in one direction: on every answered record of the Verity "
        "runs above, it finds that every check passes, as the agent did. It cannot check "
        "a refused record, which keeps no draft. Generated from "
        f"`{_rel(OUT)}` (`scripts/gate_ablation.py --check`).",
    ]
    return "\n".join(lines)


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
    report = analyse(default_settings())
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    doc = DOC.read_text(encoding="utf-8")
    rendered = _replace(doc, render(report))
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
