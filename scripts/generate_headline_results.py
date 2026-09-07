"""Render the README's "measured results" section from the committed live
scorecard JSON — never hand-typed, never transcription-drift-prone.

Reads ``scorecards/live-v1.0.0.json`` (produced by
``verity eval run --judge live --scorecard-json <path>``) and replaces the
content between two Markdown comment markers in ``README.md``:

    <!-- RESULTS:START -->
      ... regenerated table + provenance line ...
    <!-- RESULTS:END -->

Idempotent: running twice yields the same file byte-for-byte (assuming the JSON
hasn't changed). Fails LOUDLY if the markers are missing or the JSON is
malformed — the whole point is to prevent a silent stale README.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORECARD = REPO_ROOT / "scorecards" / "live-v1.0.0.json"
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


def render_block(data: dict[str, Any]) -> str:
    """Return the Markdown block to insert between the RESULTS markers."""
    prov = data["provenance"]
    cost = data["cost"]
    retr = data["retrieval"]
    ans = data["answer_quality"]
    refu = data["refusal_calibration"]
    lat = data["latency_ms"]
    ds = data["dataset"]
    is_stub_agent = prov["agent_model"] == "stub-agent"
    is_stub_judge = prov["judge_model"] == "stub-judge-v1"
    stub_warning = (
        "\n> ⚠️ **This scorecard was generated on stubs**. "
        "The v1.0.0 file will be regenerated from the one-shot live run — "
        "faithfulness / citation_accuracy / hallucination_rate below come from "
        "the mechanical stub judge (`stub-judge-v1`) and are placeholders here.\n"
        if (is_stub_agent or is_stub_judge)
        else ""
    )
    date = _short_date(str(data["generated_at"]))
    lines: list[str] = []
    lines.append(
        f"**Measured on**: `agent={prov['agent_model']}` · "
        f"`judge={prov['judge_model']}` · "
        f"`embedding={prov['embedding_model']}` · "
        f"commit `{prov['run_sha']}` · {date} · "
        f"n={ds['n_questions']} ({ds['n_answerable']} answerable + {ds['n_unanswerable']} out-of-scope) · "
        f"total cost **${cost['total_usd']:.4f}**"
    )
    lines.append("")
    lines.append("| Metric | Value | Notes |")
    lines.append("|---|---:|---|")
    lines.append(
        f"| **Refusal recall** | **{_fmt(refu['refusal_recall'])}** | headline · fraction of out-of-scope questions correctly refused |"
    )
    lines.append(
        f"| **Refusal precision** | **{_fmt(refu['refusal_precision'])}** | of the refusals, fraction that were genuinely unanswerable |"
    )
    lines.append(
        f"| Faithfulness | {_fmt(ans['faithfulness'])} | claim-level, judge=`{prov['judge_model']}` |"
    )
    lines.append(
        f"| Citation accuracy | {_fmt(ans['citation_accuracy'])} | judge=`{prov['judge_model']}` |"
    )
    lines.append(
        f"| Hallucination rate (answerable only) | {_fmt(ans['hallucination_rate_answerable'])} | ↓ better · gated on `example.answerable` |"
    )
    ooa = refu.get("out_of_scope_answered", {})
    if isinstance(ooa, dict) and "count" in ooa and "total" in ooa:
        lines.append(
            f"| Out-of-scope answered | {ooa['count']}/{ooa['total']} | ↓ better · questions we shipped an answer for that were unanswerable |"
        )
    lines.append(
        f"| Retrieval nDCG@{retr['k']} | {_fmt(retr['ndcg_at_k'])} | deterministic (no LLM) |"
    )
    lines.append(f"| Retrieval recall@{retr['k']} | {_fmt(retr['recall_at_k'])} | deterministic |")
    lines.append(
        f"| Retrieval precision@{retr['k']} | {_fmt(retr['precision_at_k'])} | k>gold ceiling; interpret with recall |"
    )
    lines.append(
        f"| Latency p50 / p95 / p99 | {_fmt_ms(lat['p50'])} / {_fmt_ms(lat['p95'])} / {_fmt_ms(lat['p99'])} | warm; cold_start `{_fmt_ms(lat.get('cold_start'))}` (isolated) |"
    )
    lines.append(
        f"| Cost / query | ${cost['per_query_usd']:.4f} | total ${cost['total_usd']:.4f} = agent ${cost['agent_usd']:.4f} + judge ${cost['judge_usd']:.4f} |"
    )
    lines.append("")
    lines.append(
        f"Numbers are copied verbatim from [`{DEFAULT_SCORECARD.relative_to(REPO_ROOT)}`]"
        f"({DEFAULT_SCORECARD.relative_to(REPO_ROOT).as_posix()}) "
        "(committed artefact of the single live run of record — never averaged, "
        "never re-run to nudge a number)."
    )
    lines.append(stub_warning)
    return "\n".join(lines)


def patch_readme(readme_path: Path, block: str) -> bool:
    """Replace the content between the RESULTS markers. Returns True if the
    file changed, False if it was already up to date."""
    text = readme_path.read_text(encoding="utf-8")
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
        return False
    # Collapse any accidental double blank lines the replacement introduced.
    replacement = re.sub(r"\n{3,}", "\n\n", replacement)
    readme_path.write_text(replacement, encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorecard", type=Path, default=DEFAULT_SCORECARD)
    parser.add_argument("--readme", type=Path, default=DEFAULT_README)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the README would be modified (CI use).",
    )
    args = parser.parse_args(argv)

    if not args.scorecard.is_file():
        raise SystemExit(f"scorecard JSON not found: {args.scorecard}")
    payload = json.loads(args.scorecard.read_text(encoding="utf-8"))
    if payload.get("schema") != "verity.scorecard.headline/v2":
        raise SystemExit(
            f"scorecard schema is not verity.scorecard.headline/v2 (got {payload.get('schema')!r})"
        )
    block = render_block(payload)
    changed = patch_readme(args.readme, block)
    if args.check and changed:
        print(
            "README would be modified — run scripts/generate_headline_results.py to regenerate.",
            file=sys.stderr,
        )
        return 1
    if changed:
        print(f"Updated {args.readme}")
    else:
        print(f"README already up to date ({args.readme})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
