"""Write the headline scorecard from a run artefact, and the README results block from
the scorecard. Nothing in either is typed by hand.

    # reproduce: artefact -> scorecard -> README
    uv run python scripts/generate_headline_results.py \
        --artefact datasets/eval/runs/live/<run>.json --scorecard scorecards/<name>.json
    # CI: scorecard == recomputed from its artefact, README block == rendered scorecard
    uv run python scripts/generate_headline_results.py --check

The scorecard names its artefact (``source_artefact``), so ``--check`` needs no
argument. The README block sits between ``<!-- RESULTS:START -->`` and
``<!-- RESULTS:END -->``; missing markers fail loudly.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from verity.eval.headline import build_headline_dict_from_eval_run
from verity.eval.run_store import load_run

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORECARD = REPO_ROOT / "scorecards" / "live-v1.1.0.json"
DEFAULT_README = REPO_ROOT / "README.md"
START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"


def _fmt(value: float | None, *, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _fmt_ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f} ms"


def _short_date(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%Y-%m-%d")
    except ValueError:
        return iso.split("T", 1)[0]


def _repo_path(path: Path) -> str:
    """``path`` relative to the repository when inside it, else absolute."""
    resolved = path.resolve()
    inside = resolved.is_relative_to(REPO_ROOT)
    return (resolved.relative_to(REPO_ROOT) if inside else resolved).as_posix()


def _ci(k: int, n: int, ci: list[float] | None) -> str:
    """``k/n · 95 % CI [low, high]`` (Wilson), or ``k/n`` when n is 0."""
    if ci is None:
        return f"{k}/{n}"
    return f"{k}/{n} · 95 % CI [{ci[0]:.3f}, {ci[1]:.3f}]"


def _usd(value: float | None) -> str:
    return "not recorded" if value is None else f"${value:.4f}"


def render_block(data: dict[str, Any], *, scorecard_path: Path = DEFAULT_SCORECARD) -> str:
    """Return the Markdown block to insert between the RESULTS markers."""
    prov = data["provenance"]
    cost = data["cost"]
    retr = data["retrieval"]
    ans = data["answer_quality"]
    refu = data["refusal"]
    lat = data["latency_ms"]
    ds = data["dataset"]
    sizes = data["sample_sizes"]
    is_stub = "stub" in prov["agent_model"] or "stub" in prov["judge_model"]
    stub_warning = (
        "\n> ⚠️ **This scorecard was generated on stubs**: the answer-quality and refusal "
        "numbers describe the scripted stubs, not Verity.\n"
        if is_stub
        else ""
    )
    date = _short_date(str(data["generated_at"]))
    without_claims = sizes["n_judged_without_claims"]
    recall, precision = refu["refusal_recall"], refu["refusal_precision"]
    thr = refu.get("confidence_threshold")
    threshold = "threshold not recorded" if thr is None else f"threshold {thr:g}"
    lines = [
        f"**Measured on**: `agent={prov['agent_model']}` · "
        f"`judge={prov['judge_model']}` · "
        f"`embedding={prov['embedding_model']}` · "
        f"commit `{prov['run_sha']}` · {date} · dataset `{ds['name']}` · "
        f"n={ds['n_questions']} ({ds['n_answerable']} answerable + "
        f"{ds['n_unanswerable']} unanswerable) · "
        f"**{sizes['n_errored']} errored** (kept out of every metric) · "
        f"total cost **{_usd(cost['total_usd'])}**",
        "",
        "| Metric | Value | Notes |",
        "|---|---:|---|",
        f"| **Refusal recall** | **{_fmt(recall['value'])}** | "
        f"unanswerable questions refused / unanswerable: "
        f"{_ci(recall['k'], recall['n'], recall['ci95_wilson'])} · {threshold} |",
        f"| **Refusal precision** | **{_fmt(precision['value'])}** | "
        f"refusals that were unanswerable / refusals: "
        f"{_ci(precision['k'], precision['n'], precision['ci95_wilson'])} |",
    ]
    for label, key in (
        ("Faithfulness", "faithfulness"),
        ("Citation accuracy", "citation_accuracy"),
    ):
        macro, micro = ans[key]["macro"], ans[key]["micro"]
        lines.append(
            f"| {label}, per question | {_fmt(macro['value'])} | "
            f"mean over {macro['n_questions']} judged answers · judge unvalidated |"
        )
        lines.append(
            f"| {label}, per claim | {_fmt(micro['value'])} | pooled over claims: "
            f"{_ci(micro['k'], micro['n_claims'], micro['ci95_wilson'])} · judge unvalidated |"
        )
    hall = ans["hallucination_rate"]
    lines.append(
        f"| Hallucination rate | {_fmt(hall['value'])} | ↓ better · answers with an "
        f"unsupported claim / answered answerable questions: "
        f"{_ci(hall['k'], hall['n_questions'], hall['ci95_wilson'])} |"
    )
    ooa = refu["out_of_scope_answered"]
    lines.append(
        f"| Unanswerable answered | {ooa['count']}/{ooa['total']} | ↓ better · "
        f"an answer shipped where the corpus has none: "
        f"{_ci(ooa['count'], ooa['total'], ooa['ci95_wilson'])} |"
    )
    lines.append(
        f"| Retrieval nDCG@{retr['k']} | {_fmt(retr['ndcg_at_k'])} | deterministic (no LLM) |"
    )
    lines.append(f"| Retrieval recall@{retr['k']} | {_fmt(retr['recall_at_k'])} | deterministic |")
    lines.append(
        f"| Retrieval precision@{retr['k']} | {_fmt(retr['precision_at_k'])} | "
        "k>gold ceiling; interpret with recall |"
    )
    cold_start = lat.get("cold_start")
    latency_note = (
        "no warmup: includes the first question's model loading"
        if cold_start is None
        else f"warm; cold start `{_fmt_ms(cold_start)}` measured separately"
    )
    lines.append(
        f"| Latency p50 / p95 / p99 | {_fmt_ms(lat['p50'])} / {_fmt_ms(lat['p95'])} / "
        f"{_fmt_ms(lat['p99'])} | {latency_note} |"
    )
    lines.append(
        f"| Cost / query | ${cost['per_query_usd']:.4f} | total {_usd(cost['total_usd'])} = "
        f"agent {_usd(cost['agent_usd'])} + judge {_usd(cost['judge_usd'])} |"
    )
    lines.append("")
    by_reason = refu.get("by_reason") or {}
    recorded = {k: v for k, v in by_reason.items() if k != "not_recorded"}
    if recorded:
        causes = " · ".join(
            f"`{k}` {v['answerable'] + v['unanswerable']} ({v['answerable']} answerable)"
            for k, v in recorded.items()
        )
        lines.append(f"Refusals by cause: {causes}.")
        if "below_threshold" not in recorded:
            lines.append(
                "No refusal has the cause `below_threshold`: the confidence threshold "
                "decided none of them."
            )
        lines.append("")
    if "not_recorded" in by_reason:
        n = sum(by_reason["not_recorded"].values())
        lines.append(f"{n} refusal(s) with no recorded cause (artefact predates it).")
        lines.append("")
    if without_claims:
        lines.append(
            f"{without_claims} judged answer(s) got no usable claim from the judge (output "
            "unparseable, or no claims): scored 0 per question, "
            "no claims counted per claim."
        )
        lines.append("")
    rel = _repo_path(scorecard_path)
    lines.append(
        f"Generated from [`{rel}`]({rel}), itself computed from the run artefact "
        f"`{data['source_artefact']}` (`scripts/generate_headline_results.py --check` "
        "recomputes both)."
    )
    lines.append(stub_warning)
    return "\n".join(lines)


def render_readme(text: str, block: str) -> str:
    """Return ``text`` with the content between the RESULTS markers replaced."""
    start = text.find(START)
    end = text.find(END)
    if start == -1 or end == -1 or end < start:
        raise SystemExit(
            f"README missing {START!r} / {END!r} markers — "
            "add them where you want the results table."
        )
    # Preserve marker lines.
    prefix = text[: start + len(START)]
    suffix = text[end:]
    replacement = f"{prefix}\n{block}\n{suffix}"
    if replacement == text:
        return text
    # Collapse any accidental double blank lines the replacement introduced.
    return re.sub(r"\n{3,}", "\n\n", replacement)


def _dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorecard", type=Path, default=DEFAULT_SCORECARD)
    parser.add_argument("--readme", type=Path, default=DEFAULT_README)
    parser.add_argument(
        "--artefact",
        type=Path,
        help="Run artefact to compute the scorecard from (default: the one it names).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if the scorecard or the README would change; write nothing.",
    )
    args = parser.parse_args(argv)

    existing = (
        json.loads(args.scorecard.read_text(encoding="utf-8")) if args.scorecard.is_file() else None
    )
    if args.artefact is not None:
        artefact = args.artefact
    elif existing is not None and existing.get("source_artefact"):
        artefact = REPO_ROOT / existing["source_artefact"]
    else:
        raise SystemExit(f"no --artefact given and {args.scorecard} names none")
    source = _repo_path(artefact)
    payload = build_headline_dict_from_eval_run(load_run(artefact), source_artefact=source)

    scorecard_text = _dump(payload)
    current = args.readme.read_text(encoding="utf-8")
    rendered = render_readme(current, render_block(payload, scorecard_path=args.scorecard))
    stale = []
    if existing is None or _dump(existing) != scorecard_text:
        stale.append(str(args.scorecard))
    if rendered != current:
        stale.append(str(args.readme))
    # --check only compares: a CI or pre-commit run must never edit the tree.
    if args.check:
        if stale:
            print(f"out of date: {', '.join(stale)} — run without --check", file=sys.stderr)
            return 1
        print(f"scorecard and README match {source}")
        return 0
    args.scorecard.parent.mkdir(parents=True, exist_ok=True)
    args.scorecard.write_text(scorecard_text, encoding="utf-8")
    args.readme.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.scorecard} and {args.readme} from {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
