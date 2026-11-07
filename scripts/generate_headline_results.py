"""Write the headline scorecard from a run artefact, the results table, the README's key
result and the system card's measured-performance block from the scorecard. Nothing is
typed by hand.

    # reproduce: artefact -> scorecard -> results doc, README, system card
    uv run python scripts/generate_headline_results.py \
        --artefact datasets/eval/runs/live/<run>.json --scorecard scorecards/<name>.json
    # CI: scorecard == recomputed from its artefact, every block == rendered scorecard
    uv run python scripts/generate_headline_results.py --check

The scorecard names its artefact (``source_artefact``), so ``--check`` needs no
argument. The results table sits between ``<!-- RESULTS:START -->`` and
``<!-- RESULTS:END -->`` in ``docs/results.md``; the README's key result between
``<!-- KEY-RESULT:START -->`` and ``<!-- KEY-RESULT:END -->``; the system card block
between ``<!-- SYSTEM-CARD:START -->`` and ``<!-- SYSTEM-CARD:END -->`` in
``docs/SYSTEM_CARD.md`` (each proportion with k/n and a Wilson 95 % interval, and the
approved SLOs next to their measured values). Missing markers fail loudly, and every
file is rendered before any is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from verity.eval.harness import percentile
from verity.eval.headline import SCHEMA, build_headline_dict_from_eval_run
from verity.eval.intervals import wilson
from verity.eval.run_store import load_run

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "datasets" / "eval" / "runs" / "live"
DEFAULT_SCORECARD = REPO_ROOT / "scorecards" / "live-v1.1.0.json"
DEFAULT_README = REPO_ROOT / "README.md"
DEFAULT_RESULTS_DOC = REPO_ROOT / "docs" / "results.md"
DEFAULT_SYSTEM_CARD = REPO_ROOT / "docs" / "SYSTEM_CARD.md"
VALIDATION = REPO_ROOT / "datasets" / "eval" / "judge_validation"
START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"
KEY_START = "<!-- KEY-RESULT:START -->"
KEY_END = "<!-- KEY-RESULT:END -->"
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


def _repo_path(path: Path) -> str:
    """``path`` relative to the repository when inside it, else absolute."""
    resolved = path.resolve()
    inside = resolved.is_relative_to(REPO_ROOT)
    return (resolved.relative_to(REPO_ROOT) if inside else resolved).as_posix()


def _link(target: Path, doc: Path) -> str:
    """Markdown link target for ``target`` as seen from the file ``doc``."""
    return Path(os.path.relpath(target.resolve(), doc.resolve().parent)).as_posix()


def _ci(k: int, n: int, ci: list[float] | None) -> str:
    """``k/n · 95 % CI [low, high]`` (Wilson), or ``k/n`` when n is 0."""
    if ci is None:
        return f"{k}/{n}"
    return f"{k}/{n} · 95 % CI {_bracket(ci)}"


def _usd(value: float | None) -> str:
    return "not recorded" if value is None else f"${value:.4f}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def judge_label(judge_model: str, judge_prompt_sha: str, validation_dir: Path = VALIDATION) -> str:
    """How the answer-quality rows qualify their judge: validated only by a blind
    agreement artefact measured on this very judge model *and* judge prompt. A κ
    measured on one prompt version says nothing about another."""
    v010 = validation_dir / "judge_faithfulness.v0.1.0.prompt"
    agreement, rejudged = (
        validation_dir / "agreement.json",
        validation_dir / "perturbed_verdicts.json",
    )
    model = judge_model.split()[0]
    if v010.is_file() and _sha256(v010) == judge_prompt_sha:
        if not (agreement.is_file() and rejudged.is_file()):
            return "judge unvalidated"
        if model != json.loads(rejudged.read_text(encoding="utf-8"))["judge_model"]:
            return "judge unvalidated"
        mixed = json.loads(agreement.read_text(encoding="utf-8"))["agreement_mixed"]
        label = f"judge κ {mixed['cohen_kappa']:.2f}"
        kappa_ci = validation_dir / "kappa_ci.json"
        if not kappa_ci.is_file():
            return f"{label} vs blind human (n={mixed['n_claims']})"
        ci = json.loads(kappa_ci.read_text(encoding="utf-8"))
        low, high = ci["all_claims"]["ci95_percentile"]
        real = ci["n_real_unsupported_by_human"]
        return (
            f"{label} (bootstrap 95 % CI [{low:.2f}, {high:.2f}]) vs blind human, validated on "
            f"{mixed['n_claims']} claims of which {real} real hallucination"
            f"{'' if real == 1 else 's'}"
        )
    v020 = validation_dir / "judge-v0.2.0"
    verdicts = v020 / "verdicts.json"
    if verdicts.is_file() and agreement.is_file():
        meta = json.loads(verdicts.read_text(encoding="utf-8"))
        if meta["prompt_sha256"] == judge_prompt_sha and meta["judge_model"] == model:
            mixed = json.loads((v020 / "agreement.json").read_text(encoding="utf-8"))
            total = json.loads(agreement.read_text(encoding="utf-8"))["agreement_mixed"]
            return (
                f"judge prompt v0.2.0 κ {mixed['agreement_mixed']['cohen_kappa']:.2f} vs blind "
                f"human on {mixed['agreement_mixed']['n_claims']} of {total['n_claims']} "
                "claims (its rejudge stopped at its spending cap; no bootstrap interval)"
            )
    return "judge unvalidated"


def _completed(data: dict[str, Any]) -> str:
    """``n = 56``, or how many a run cut short (spending cap, outage) completed."""
    n, sizes = data["dataset"]["n_questions"], data["sample_sizes"]
    missing = sizes.get("n_errored", 0) + sizes.get("n_not_run", 0)
    if not missing:
        return f"n = {n}"
    return (
        f"n = {n - missing} of {n} completed ({sizes.get('n_not_run', 0)} not run, stopped "
        f"by the run's spending cap; {sizes.get('n_errored', 0)} errored)"
    )


def _judged_of_shipped(data: dict[str, Any]) -> str:
    sizes = data["sample_sizes"]
    judged, answered = sizes["n_judged"], sizes["n_answered"]
    if judged == answered:
        return f"all {answered} shipped answers were judged"
    return f"{judged} of {answered} shipped answers were judged"


def _per_query_with_judge(cost: dict[str, Any]) -> str:
    value = cost.get("per_query_with_judge_usd")
    return "with the judge not recorded" if value is None else f"${value:.4f} with the judge"


def _invented_row(invented: dict[str, Any], judge: str) -> str:
    ids = ", ".join(f"`{i}`" for i in invented["example_ids"]) or "none"
    return (
        f"| **Invented answers** | **{invented['k']}/{invented['n_questions']}** | ↓ better · "
        "answers with a claim the judge finds unsupported (or no usable claim) / every "
        f"judged answer, answerable or not: {_ci(invented['k'], invented['n_questions'], invented['ci95_wilson'])}"
        f" · {ids} · {judge} |"
    )


def render_block(
    data: dict[str, Any],
    *,
    scorecard_path: Path = DEFAULT_SCORECARD,
    doc_path: Path = DEFAULT_RESULTS_DOC,
    judge: str = "judge unvalidated",
) -> str:
    """Return the Markdown block to insert between the RESULTS markers of ``doc_path``."""
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
        f"**{sizes['n_errored']} errored, {sizes.get('n_not_run', 0)} not run** (kept out of "
        "every metric) · "
        f"total cost **{_usd(cost['total_usd'])}**",
        "",
        "| Metric | Value | Notes |",
        "|---|---:|---|",
        _invented_row(ans["invented_answer_rate"], judge),
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
            f"mean over {macro['n_questions']} judged answers · {judge} |"
        )
        lines.append(
            f"| {label}, per claim | {_fmt(micro['value'])} | pooled over claims: "
            f"{_ci(micro['k'], micro['n_claims'], micro['ci95_wilson'])} · {judge} |"
        )
    hall = ans["hallucination_rate"]
    lines.append(
        f"| Hallucination rate, answerable only | {_fmt(hall['value'])} | ↓ better · answers "
        f"with an unsupported claim / answered answerable questions (an answer to an "
        f"unanswerable question is outside it): "
        f"{_ci(hall['k'], hall['n_questions'], hall['ci95_wilson'])} |"
    )
    ooa = refu["out_of_scope_answered"]
    lines.append(
        f"| Unanswerable answered | {ooa['count']}/{ooa['total']} | ↓ better · "
        f"an answer shipped where the corpus has none: "
        f"{_ci(ooa['count'], ooa['total'], ooa['ci95_wilson'])} |"
    )
    lines.append(
        f"| Retrieval nDCG@{retr['k']} | {_fmt(retr['ndcg_at_k'])} | "
        "scored without an LLM on hits retrieved for the LLM sub-questions |"
    )
    lines.append(f"| Retrieval recall@{retr['k']} | {_fmt(retr['recall_at_k'])} | as nDCG |")
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
        f"| Cost / query | ${cost['per_query_usd']:.4f} | ${cost['per_query_usd']:.4f} agent only"
        f" · {_per_query_with_judge(cost)} · total {_usd(cost['total_usd'])} = "
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
        f"Generated from [`{rel}`]({_link(scorecard_path, doc_path)}), itself computed from the run artefact "
        f"`{data['source_artefact']}` (`scripts/generate_headline_results.py --check` "
        "recomputes both)."
    )
    lines.append(stub_warning)
    return "\n".join(lines)


def _require_schema(data: dict[str, Any]) -> None:
    """The card reads the counts the current scorecard schema stores; an older one does
    not hold them all."""
    if data.get("schema") != SCHEMA:
        raise SystemExit(f"the system card renders {SCHEMA} scorecards, not {data.get('schema')!r}")


def _count_and_interval(k: int | None, n: int) -> tuple[str, str]:
    """``("k/n", "[low, high]")``, or ``("n=<n>", "no interval")`` when k is unknown."""
    interval = wilson(k, n) if k is not None else None
    if k is None or interval is None:
        return f"n={n}", "no interval"
    return f"{k}/{n}", f"[{interval[0]:.3f}, {interval[1]:.3f}]"


def _ci_cells(k: int | None, n: int) -> str:
    return " | ".join(_count_and_interval(k, n))


def _performance_rows(data: dict[str, Any]) -> list[str]:
    _require_schema(data)
    refusal = data["refusal"]
    rows: list[str] = []
    for label, name in (
        ("Refusal recall", "refusal_recall"),
        ("Refusal precision", "refusal_precision"),
    ):
        metric = refusal[name]
        rows.append(
            f"| {label} | {_fmt(metric['value'])} | {_ci_cells(metric['k'], metric['n'])} |"
        )
    ooa = refusal["out_of_scope_answered"]
    ooa_value = ooa["count"] / ooa["total"] if ooa["total"] else 0.0
    rows.append(
        f"| Out-of-scope answered ↓ | {_fmt(ooa_value)} | {_ci_cells(ooa['count'], ooa['total'])} |"
    )
    invented = data["answer_quality"]["invented_answer_rate"]
    rows.append(
        f"| Invented answers, every answer shipped ↓ (judge) | {_fmt(invented['value'])} | "
        f"{_ci_cells(invented['k'], invented['n_questions'])} |"
    )
    hall = data["answer_quality"]["hallucination_rate"]
    rows.append(
        f"| Hallucination rate, answerable only ↓ (judge) | {_fmt(hall['value'])} | "
        f"{_ci_cells(hall['k'], hall['n_questions'])} |"
    )
    for label, name in (
        ("Faithfulness", "faithfulness"),
        ("Citation accuracy", "citation_accuracy"),
    ):
        micro = data["answer_quality"][name]["micro"]
        rows.append(
            f"| {label} per claim (judge) | {_fmt(micro['value'])} | "
            f"{_ci_cells(micro['k'], micro['n_claims'])} |"
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


def _against_proportion(k: int, n: int, target: float, cause: str) -> str:
    """A rate meets its target only when its Wilson upper bound does: 0 errors in 56
    requests is no violation observed, not a rate of 1 % or less. The test is one-sided
    on purpose: a measured rate over target is a miss on the run of record whatever its
    interval, as the latency SLO reads its measured p95."""
    interval = wilson(k, n)
    if interval is None:
        return "not measured"
    if k / n > target:
        return f"**SLO not met**: {cause}"
    if interval[1] > target:
        seen = "no violation observed" if k == 0 else f"{k} of {n} within target"
        return (
            f"{seen}; n too small to conclude (95 % upper bound {interval[1]:.1%} > {target:.0%})"
        )
    return "met"


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
    # Questions the eval's own spending cap left unrun were never requested.
    n = data["dataset"]["n_questions"] - data["sample_sizes"].get("n_not_run", 0)
    errored = int(data["sample_sizes"]["n_errored"])
    error_rate = errored / n if n else None
    count, interval = _count_and_interval(errored, n)
    error_cell = f"{_fmt(error_rate)} ({count}, 95 % CI {interval})"
    per_query = float(data["cost"]["per_query_usd"])
    approved = "approved"
    return [
        *_latency_slo_rows(data, approved),
        f"| Error rate (requests answered with neither an answer nor a refusal) | "
        f"≤ {SLO_ERROR_RATE:.0%} ({approved}) | "
        f"{error_cell} | "
        + _against_proportion(
            errored,
            n,
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
    link = _link(scorecard_path, card_path)
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


def render_key_result(
    data: dict[str, Any],
    *,
    readme_path: Path = DEFAULT_README,
    results_doc: Path = DEFAULT_RESULTS_DOC,
    judge: str = "judge unvalidated",
) -> str:
    """Return the README's key result: invented answers over every answer shipped, then
    refusal recall and precision, each with its interval."""
    prov, ds, refu = data["provenance"], data["dataset"], data["refusal"]
    recall, precision = refu["refusal_recall"], refu["refusal_precision"]
    invented = data["answer_quality"]["invented_answer_rate"]
    ids = ", ".join(f"`{i}`" for i in invented["example_ids"])
    stub = "stub" in prov["agent_model"] or "stub" in prov["judge_model"]
    lines = [
        f"**Invented answers {invented['k']}/{invented['n_questions']}** "
        f"(Wilson 95 % CI {_bracket(invented['ci95_wilson'])}): answers holding a claim the "
        "judge finds unsupported by the retrieved passages, over every judged answer, "
        f"answerable or not{f' ({ids})' if ids else ''}; {_judged_of_shipped(data)}. "
        "Faithfulness to the passages; correctness against the reference answers is "
        "graded apart in the results. "
        f"{judge[0].upper()}{judge[1:]}.",
        f"**Refusal recall {_fmt(recall['value'])}**: "
        f"{recall['k']}/{recall['n']} unanswerable questions refused, Wilson 95 % CI "
        f"{_bracket(recall['ci95_wilson'])}, at refusal precision {_fmt(precision['value'])} "
        f"({precision['k']}/{precision['n']}, CI {_bracket(precision['ci95_wilson'])}).",
        f"One live run on the {_split_label(ds['name'])}, {_completed(data)}, "
        f"agent `{prov['agent_model'].split()[0]}`, commit `{prov['run_sha']}`. "
        f"Every metric, the judge's validation and the limits: "
        f"[`docs/results.md`]({_link(results_doc, readme_path)}).",
    ]
    if stub:
        lines.append("")
        lines.append("> ⚠️ **Generated on stubs**: this describes the scripted stubs, not Verity.")
    return "\n".join(lines)


def _split_label(name: str) -> str:
    # Only the test split is held out: dev chose the threshold, so calling it held-out would overstate.
    return "held-out `test` split" if name == "test" else f"`{name}` dataset"


def _bracket(ci: list[float] | None) -> str:
    return "n/a" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def render_results(text: str, block: str) -> str:
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


def _dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorecard", type=Path, default=DEFAULT_SCORECARD)
    parser.add_argument("--readme", type=Path, default=DEFAULT_README)
    parser.add_argument("--results-doc", type=Path, default=DEFAULT_RESULTS_DOC)
    parser.add_argument("--system-card", type=Path, default=DEFAULT_SYSTEM_CARD)
    parser.add_argument(
        "--artefact",
        type=Path,
        help="Run artefact to compute the scorecard from (default: the one it names).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if the scorecard or a generated block would change; write nothing.",
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
    for path in (args.readme, args.results_doc, args.system_card):
        if not path.is_file():
            raise SystemExit(f"{path} not found")
    readme = args.readme.read_text(encoding="utf-8")
    results = args.results_doc.read_text(encoding="utf-8")
    card = args.system_card.read_text(encoding="utf-8")
    # Render every file before writing any, so a missing marker leaves all as they were.
    judge = judge_label(
        payload["provenance"]["judge_model"],
        # No recorded judge prompt (a stub run) leaves the judge unvalidated.
        payload["provenance"]["prompt_sha256"].get("judge_faithfulness", ""),
    )
    rendered_results = render_results(
        results,
        render_block(
            payload, scorecard_path=args.scorecard, doc_path=args.results_doc, judge=judge
        ),
    )
    rendered_readme = replace_between(
        readme,
        render_key_result(
            payload, readme_path=args.readme, results_doc=args.results_doc, judge=judge
        ),
        KEY_START,
        KEY_END,
    )
    rendered_card = replace_between(
        card,
        render_system_card_block(
            payload, scorecard_path=args.scorecard, card_path=args.system_card
        ),
        CARD_START,
        CARD_END,
    )
    pending: list[tuple[Path, str]] = []
    if existing is None or _dump(existing) != scorecard_text:
        pending.append((args.scorecard, scorecard_text))
    if rendered_results != results:
        pending.append((args.results_doc, rendered_results))
    if rendered_readme != readme:
        pending.append((args.readme, rendered_readme))
    if rendered_card != card:
        pending.append((args.system_card, rendered_card))
    # --check only compares: a CI or pre-commit run must never edit the tree.
    if args.check:
        if pending:
            names = ", ".join(str(path) for path, _ in pending)
            print(f"out of date: {names} — run without --check", file=sys.stderr)
            return 1
        print(f"scorecard, results doc, README and system card match {source}")
        return 0
    args.scorecard.parent.mkdir(parents=True, exist_ok=True)
    for path, text in pending:
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
