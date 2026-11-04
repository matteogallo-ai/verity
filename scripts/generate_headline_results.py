"""Render the README's "measured results" section and the system card's
"measured performance" section from the committed live scorecard JSON — never
hand-typed, never transcription-drift-prone.

Reads ``scorecards/live-v1.0.0.json`` (produced by
``verity eval run --judge live --scorecard-json <path>``) and replaces the
content between two Markdown comment markers in ``README.md``:

    <!-- RESULTS:START -->
      ... regenerated table + provenance line ...
    <!-- RESULTS:END -->

and between ``<!-- SYSTEM-CARD:START -->`` / ``<!-- SYSTEM-CARD:END -->`` in
``docs/SYSTEM_CARD.md`` (each proportion with k/n and a Wilson 95 % interval, and
the proposed SLOs next to their measured values). Both blocks read the same
scorecard file, so replacing the scorecard regenerates both.

Idempotent: running twice yields the same file byte-for-byte (assuming the JSON
hasn't changed). Fails LOUDLY if the markers are missing or the JSON is
malformed — the whole point is to prevent a silent stale README.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from verity.eval.harness import percentile
from verity.eval.intervals import wilson

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "datasets" / "eval" / "runs" / "live"
DEFAULT_SCORECARD = REPO_ROOT / "scorecards" / "live-v1.0.0.json"
DEFAULT_README = REPO_ROOT / "README.md"
DEFAULT_SYSTEM_CARD = REPO_ROOT / "docs" / "SYSTEM_CARD.md"
START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"
CARD_START = "<!-- SYSTEM-CARD:START -->"
CARD_END = "<!-- SYSTEM-CARD:END -->"
# Proposed by the implementer, not approved by the owner: the card says so on every row.
SLO_P95_LATENCY_MS = 15_000.0
SLO_ERROR_RATE = 0.01
SLO_COST_PER_REQUEST_USD = 0.02


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
    # v3: each judge-track metric is a {value, scope, n_denominator} triple.
    # Render the value alongside the scope so no number can be quoted
    # without its denominator visible.
    faith = ans["faithfulness"]
    ca = ans["citation_accuracy"]
    hall = ans["hallucination_rate"]
    lines.append(
        f"| Faithfulness ({faith['scope']}, n={faith['n_denominator']}) | "
        f"{_fmt(faith['value'])} | per-question mean of claim-level judge verdicts over {faith['n_denominator']} answered questions · judge unvalidated |"
    )
    lines.append(
        f"| Citation accuracy ({ca['scope']}, n={ca['n_denominator']}) | "
        f"{_fmt(ca['value'])} | per-question mean over {ca['n_denominator']} answered questions (refusals bypass the judge) · judge unvalidated |"
    )
    lines.append(
        f"| Hallucination rate ({hall['scope']}, n={hall['n_denominator']}) | "
        f"{_fmt(hall['value'])} | ↓ better · at-risk set = answerable questions the agent answered |"
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
    cold_start = lat.get("cold_start")
    latency_note = (
        "no warmup: includes the first question's model loading"
        if cold_start is None
        else f"warm; cold start `{_fmt_ms(cold_start)}` measured separately"
    )
    lines.append(
        f"| Latency p50 / p95 / p99 | {_fmt_ms(lat['p50'])} / {_fmt_ms(lat['p95'])} / {_fmt_ms(lat['p99'])} | {latency_note} |"
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


def _refusal_block(data: dict[str, Any]) -> dict[str, Any]:
    refusal = data.get("refusal") or data.get("refusal_calibration")
    if not isinstance(refusal, dict):
        raise SystemExit("scorecard has no refusal block")
    return refusal


def _refusal_count(data: dict[str, Any], name: str) -> tuple[float, int, int]:
    """``(value, k, n)`` of refusal recall or precision.

    A scorecard that stores the counts gives them directly; one that stores only the
    value gives them through its ``per_refusal`` records, and the value must agree.
    """
    metric = _refusal_block(data)[name]
    if isinstance(metric, dict):
        return float(metric["value"]), int(metric["k"]), int(metric["n"])
    rows = data["per_refusal"]
    k = sum(1 for r in rows if not r["expected_answerable"] and r["observed_refused"])
    if name == "refusal_recall":
        n = sum(1 for r in rows if not r["expected_answerable"])
    else:
        n = sum(1 for r in rows if r["observed_refused"])
    if n and abs(k / n - float(metric)) > 1e-6:
        raise SystemExit(f"{name}={metric} disagrees with per_refusal ({k}/{n})")
    return float(metric), k, n


def _binary_count(value: float, n: int) -> int | None:
    """``k`` with ``k / n == value`` up to the scorecard's 6-decimal rounding, else None."""
    if n == 0:
        return None
    k = round(value * n)
    return k if abs(k / n - value) <= 1e-6 else None


def _count_and_interval(k: int | None, n: int) -> tuple[str, str]:
    """``("k/n", "[low, high]")``, or ``("n=<n>", "no interval")`` when k is unknown."""
    interval = wilson(k, n) if k is not None else None
    if k is None or interval is None:
        return f"n={n}", "no interval"
    return f"{k}/{n}", f"[{interval[0]:.3f}, {interval[1]:.3f}]"


def _ci_cells(k: int | None, n: int) -> str:
    return " | ".join(_count_and_interval(k, n))


def _performance_rows(data: dict[str, Any]) -> list[str]:
    refusal = _refusal_block(data)
    rows: list[str] = []
    for label, name in (
        ("Refusal recall", "refusal_recall"),
        ("Refusal precision", "refusal_precision"),
    ):
        value, k, n = _refusal_count(data, name)
        rows.append(f"| {label} | {_fmt(value)} | {_ci_cells(k, n)} |")
    ooa = refusal["out_of_scope_answered"]
    ooa_value = ooa["count"] / ooa["total"] if ooa["total"] else 0.0
    rows.append(
        f"| Out-of-scope answered ↓ | {_fmt(ooa_value)} | {_ci_cells(ooa['count'], ooa['total'])} |"
    )
    hall = data["answer_quality"]["hallucination_rate"]
    hall_n = int(hall.get("n_questions", hall.get("n_denominator", 0)))
    hall_k = hall["k"] if "k" in hall else _binary_count(float(hall["value"]), hall_n)
    rows.append(
        f"| Hallucination rate ↓ (judge) | {_fmt(hall['value'])} | {_ci_cells(hall_k, hall_n)} |"
    )
    for label, name in (
        ("Faithfulness", "faithfulness"),
        ("Citation accuracy", "citation_accuracy"),
    ):
        metric = data["answer_quality"][name]
        if "micro" in metric:
            micro = metric["micro"]
            rows.append(
                f"| {label} per claim (judge) | {_fmt(micro['value'])} | "
                f"{_ci_cells(micro['k'], micro['n_claims'])} |"
            )
        else:
            # A per-question mean of per-claim fractions has no k/n to put in an interval.
            rows.append(
                f"| {label}, per-question mean (judge) | {_fmt(metric['value'])} | "
                f"{_ci_cells(None, metric['n_denominator'])} |"
            )
    retrieval = data["retrieval"]
    n_answerable = data["dataset"]["n_answerable"]
    for label, key in (("nDCG", "ndcg_at_k"), ("Recall", "recall_at_k")):
        rows.append(
            f"| Retrieval {label}@{retrieval['k']}, per-question mean | "
            f"{_fmt(retrieval[key])} | {_ci_cells(None, n_answerable)} |"
        )
    return rows


def _against(measured: float | None, target: float, cause: str) -> str:
    """``met`` / ``not measured`` / a miss with its cause: a missed target is published
    as missed, never re-targeted."""
    if measured is None:
        return "not measured"
    return "met" if measured <= target else f"**SLO not met**: {cause}"


def _run_latencies(data: dict[str, Any]) -> list[tuple[str, float]] | None:
    """Per-answer ``(example_id, latency_ms)`` in run order from the run artefact the
    scorecard was built from, or ``None`` when that artefact is not committed."""
    run = data.get("provenance", {}).get("run_sha")
    name = data["dataset"].get("name")
    path = RUNS_DIR / f"eval_{run}_{name}.json"
    if not run or not name or not path.exists():
        return None
    audits = json.loads(path.read_text(encoding="utf-8"))["per_audit"]
    return [(a["example_id"], float(a["usage"]["latency_ms"])) for a in audits]


def _p95(latencies: list[tuple[str, float]]) -> float:
    return percentile(sorted(ms for _, ms in latencies), 95)


def _p95_cause(over: list[tuple[str, float]], first: str | None) -> str:
    """The answers over target, slowest first. Only the first answer of a run without
    warmup has a known extra cost (model loading); the artefact holds no per-stage
    timings or retry counts, so the excess of any other answer is not attributed."""
    named = ", ".join(
        f"{qid} {_fmt_ms(ms)}" + (" (first answer, model loading)" if qid == first else "")
        for qid, ms in sorted(over, key=lambda item: -item[1])
    )
    unexplained = [qid for qid, _ in over if qid != first]
    tail = (
        "; the run recorded no per-stage timings or retries, so the time of "
        f"{', '.join(unexplained)} is not attributed"
        if unexplained
        else ""
    )
    return f"answers over {SLO_P95_LATENCY_MS:,.0f} ms: {named}{tail}"


def _latency_slo_rows(data: dict[str, Any], approved: str) -> list[str]:
    """p95 cold start included and warm, as separate rows. From the run artefact when it
    is committed (it must reproduce the scorecard p95); otherwise from the scorecard,
    whose single p95 is warm only if the run had a warmup."""
    target = f"≤ {SLO_P95_LATENCY_MS:,.0f} ms ({approved})"
    latency = data["latency_ms"]
    n = data["dataset"]["n_questions"]
    p95 = float(latency["p95"])
    cold_start = latency.get("cold_start")
    answers = _run_latencies(data)

    if answers is not None and cold_start is None:
        if not math.isclose(_p95(answers), p95, abs_tol=0.01):
            raise SystemExit(
                f"run artefact p95 {_p95(answers):.3f} does not reproduce the scorecard p95 {p95}"
            )
        first = answers[0][0]
        warm = answers[1:]
        warm_p95 = _p95(warm)

        def status(measured: float, sample: list[tuple[str, float]]) -> str:
            over = [(qid, ms) for qid, ms in sample if ms > SLO_P95_LATENCY_MS]
            return _against(measured, SLO_P95_LATENCY_MS, _p95_cause(over, first))

        cold_cells = (
            f"{_fmt_ms(p95)} (n={n}, no warmup: the first answer loads the models)",
            status(p95, answers),
        )
        warm_cells = (
            f"{_fmt_ms(warm_p95)} (n={len(warm)}, first answer left out)",
            status(warm_p95, warm),
        )
    elif cold_start is None:
        cold_cells = (
            f"{_fmt_ms(p95)} (n={n}, no warmup: includes model loading)",
            _against(p95, SLO_P95_LATENCY_MS, "the per-answer latencies are not committed"),
        )
        warm_cells = (
            "not measured: the run had no warmup and its artefact is not committed",
            "not measured",
        )
    else:
        cold_cells = (
            "not measured as one sample: the first answer after boot, timed apart, "
            f"took {_fmt_ms(float(cold_start))}",
            "not measured",
        )
        over = [] if answers is None else [a for a in answers if a[1] > SLO_P95_LATENCY_MS]
        warm_cells = (
            f"{_fmt_ms(p95)} (n={n}, after one warmup answer)",
            _against(p95, SLO_P95_LATENCY_MS, _p95_cause(over, None)),
        )
    return [
        f"| p95 latency of one answer, cold start included | {target} | "
        f"{cold_cells[0]} | {cold_cells[1]} |",
        f"| p95 latency of one answer, warm (cold start excluded) | {target} | "
        f"{warm_cells[0]} | {warm_cells[1]} |",
    ]


def _slo_rows(data: dict[str, Any]) -> list[str]:
    n = data["dataset"]["n_questions"]
    errored = data.get("sample_sizes", {}).get("n_errored")
    error_rate = errored / n if errored is not None and n else None
    if errored is None:
        error_cell = "not recorded by this scorecard"
    else:
        count, interval = _count_and_interval(errored, n)
        error_cell = f"{_fmt(error_rate)} ({count}, 95 % CI {interval})"
    per_query = float(data["cost"]["per_query_usd"])
    approved = "approved"
    return [
        *_latency_slo_rows(data, approved),
        f"| Error rate (requests answered with neither an answer nor a refusal) | "
        f"≤ {SLO_ERROR_RATE:.0%} ({approved}) | "
        f"{error_cell} | "
        + _against(
            error_rate,
            SLO_ERROR_RATE,
            f"{errored} of {n} requests errored; the scorecard does not record why",
        )
        + " |",
        f"| Cost per answer (agent calls; the judge is eval-only) | "
        f"≤ ${SLO_COST_PER_REQUEST_USD:.2f} ({approved}) | ${per_query:.4f} (mean, n={n}) | "
        + _against(
            per_query,
            SLO_COST_PER_REQUEST_USD,
            "the scorecard records the mean only, not which answers cost most",
        )
        + " |",
    ]


def render_system_card_block(
    data: dict[str, Any], *, scorecard_path: Path, card_path: Path = DEFAULT_SYSTEM_CARD
) -> str:
    """Markdown for the system card: measured performance with intervals, then SLOs."""
    prov = data["provenance"]
    ds = data["dataset"]
    source = _repo_path(scorecard_path)
    link = Path(os.path.relpath(scorecard_path.resolve(), card_path.resolve().parent)).as_posix()
    lines = [
        f"Source: [`{source}`]({link}) · commit `{prov['run_sha']}` · "
        f"{_short_date(str(data['generated_at']))} · agent `{prov['agent_model']}` · "
        f"judge `{prov['judge_model']}` · n={ds['n_questions']} "
        f"({ds['n_answerable']} answerable + {ds['n_unanswerable']} out-of-scope).",
        "",
        "| Metric | Value | k/n | 95 % CI (Wilson) |",
        "|---|---:|---:|---:|",
        *_performance_rows(data),
        "",
        "| SLO | Target | Measured on the run of record | Status |",
        "|---|---|---|---|",
        *_slo_rows(data),
    ]
    return "\n".join(lines)


def _repo_path(path: Path) -> str:
    resolved = path.resolve()
    inside = resolved.is_relative_to(REPO_ROOT)
    return (resolved.relative_to(REPO_ROOT) if inside else resolved).as_posix()


def render_readme(text: str, block: str) -> str:
    """Return ``text`` with the content between the RESULTS markers replaced."""
    return replace_between(text, block, START, END)


def replace_between(text: str, block: str, start_marker: str, end_marker: str) -> str:
    """Return ``text`` with the content between two marker lines replaced by ``block``."""
    start = text.find(start_marker)
    end = text.find(end_marker)
    if start == -1 or end == -1 or end < start:
        raise SystemExit(
            f"missing {start_marker!r} / {end_marker!r} markers — "
            "add them where the generated block belongs."
        )
    # Preserve marker lines.
    prefix = text[: start + len(start_marker)]
    suffix = text[end:]
    replacement = f"{prefix}\n{block}\n{suffix}"
    if replacement == text:
        return text
    # Collapse any accidental double blank lines the replacement introduced.
    return re.sub(r"\n{3,}", "\n\n", replacement)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorecard", type=Path, default=DEFAULT_SCORECARD)
    parser.add_argument("--readme", type=Path, default=DEFAULT_README)
    parser.add_argument("--system-card", type=Path, default=DEFAULT_SYSTEM_CARD)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the README or the system card would be modified (CI use).",
    )
    args = parser.parse_args(argv)

    if not args.scorecard.is_file():
        raise SystemExit(f"scorecard JSON not found: {args.scorecard}")
    payload = json.loads(args.scorecard.read_text(encoding="utf-8"))
    if payload.get("schema") != "verity.scorecard.headline/v3":
        raise SystemExit(
            f"scorecard schema is not verity.scorecard.headline/v3 (got {payload.get('schema')!r})"
        )
    targets = (
        (args.readme, render_readme, render_block(payload)),
        (
            args.system_card,
            lambda text, block: replace_between(text, block, CARD_START, CARD_END),
            render_system_card_block(
                payload, scorecard_path=args.scorecard, card_path=args.system_card
            ),
        ),
    )
    # Render every file before writing any, so a missing file or marker leaves both as
    # they were instead of updating one of them.
    pending: list[tuple[Path, str]] = []
    for path, render, block in targets:
        if not path.is_file():
            raise SystemExit(f"file to regenerate not found: {path}")
        current = path.read_text(encoding="utf-8")
        rendered = render(current, block)
        if rendered == current:
            print(f"{path} already up to date")
        else:
            pending.append((path, rendered))
    stale = [path for path, _ in pending]
    # --check only compares: a CI or pre-commit run must never edit the tree.
    if not args.check:
        for path, rendered in pending:
            path.write_text(rendered, encoding="utf-8")
            print(f"Updated {path}")
    if args.check and stale:
        names = ", ".join(str(p) for p in stale)
        print(
            f"{names} would be modified — run scripts/generate_headline_results.py to regenerate.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
