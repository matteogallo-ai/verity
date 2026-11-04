"""Judge validation: blind annotation of real and perturbed claims, the judge's
sensitivity on the perturbed ones, and Cohen's κ on the mixed set.

    # 1. sample 40 real claims (every unsupported one included) and make 20 false
    #    ones (verity.eval.perturb); writes two blind CSVs of 30 and the key ($0)
    uv run python scripts/judge_agreement.py export RUN.json [RUN.json ...]
    # 2. what step 3 costs, from the judge calls recorded in the runs ($0)
    uv run python scripts/judge_agreement.py estimate
    # 3. PAID: the judge reads each perturbed claim against its answer's evidence
    uv run python scripts/judge_agreement.py rejudge --budget-usd 0.60
    # 4. a human fills `human_verdict` (supported / unsupported) in both CSVs
    # 5. sensitivity on the perturbed claims, κ on all 60, the final CSV
    uv run python scripts/judge_agreement.py kappa
    # 6. disagreements and perturbed claims judged true, with both verdicts shown,
    #    decided one by one (scripts/annotate.py --adjudicate), then measured apart;
    #    the blind labels and agreement.json are never rewritten
    uv run python scripts/judge_agreement.py adjudication-export
    uv run python scripts/judge_agreement.py adjudicated

Why perturbed claims: the judge marked 161 of the 164 live claims supported, so on a
sample of real claims its labels are almost all "supported" and κ is undefined or
rests on one or two items. A perturbed claim is
the original with one span changed (figure, date, negation, entity of the same type).

Blindness: the CSVs hold an opaque item id, the claim, the passage its answer cited
(the citation closest to the claim, first), then every extract the judge saw. They
do not say which claims are perturbed, nor any verdict; ``judge_key.json`` holds
both and must stay closed until the annotation is done. A perturbed claim and real
claims from the same answer are never both in the sample: side by side, the
disagreement would give the perturbation away. Every answer gives exactly two items,
so a repeated block of extracts says nothing about origin either.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import re
import statistics
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from verity.eval.intervals import wilson
from verity.eval.perturb import Kind, Perturbation, perturb
from verity.eval.run_store import load_run
from verity.types import (
    Answer,
    AnswerMetrics,
    Citation,
    ClaimVerdict,
    Confidence,
    EvalExample,
    PerExampleAudit,
    UsageStats,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "datasets" / "eval" / "judge_validation"
ENTITIES = REPO_ROOT / "datasets" / "eval" / "entities.json"
CSV_FILES = (OUT_DIR / "annotate_morning_1.csv", OUT_DIR / "annotate_morning_2.csv")
KEY = OUT_DIR / "judge_key.json"
REJUDGED = OUT_DIR / "perturbed_verdicts.json"
FINAL_CSV = OUT_DIR / "claims_annotated.csv"
RESULT = OUT_DIR / "agreement.json"
ADJUDICATION = OUT_DIR / "adjudication.csv"
ADJUDICATED = OUT_DIR / "agreement_adjudicated.json"
ADJ_COLUMNS = [
    "item_id",
    "claim",
    "cited_passage",
    "extracts",
    "blind_verdict",
    "judge_verdict",
    "adjudicated_verdict",
    "reason",
]
STAYS_TRUE, ANNOTATOR_ERROR = "perturbation stays true", "annotator error"

SEED = 20261102
N_REAL = 40
N_PER_KIND = 5
PER_ANSWER = 2
LABELS = ("supported", "unsupported")
COLUMNS = ["item_id", "claim", "cited_passage", "extracts", "human_verdict"]


@dataclass(frozen=True)
class Claim:
    claim_id: str
    claim: str
    judge_verdict: str
    run: str
    audit: PerExampleAudit

    @property
    def answer_id(self) -> str:
        return self.claim_id.rsplit("/", 1)[0]


def _rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def collect(runs: Sequence[Path]) -> list[Claim]:
    """Every claim the judge returned in ``runs``, with the audit it came from."""
    claims: list[Claim] = []
    for path in runs:
        for audit in load_run(path).per_audit:
            for j, verdict in enumerate(audit.judge_claims or ()):
                claims.append(
                    Claim(
                        # The run file's name, not the dataset's: two runs of one
                        # dataset must not share ids.
                        claim_id=f"{path.stem}/{audit.example_id}/{j}",
                        claim=verdict.claim,
                        judge_verdict=LABELS[0] if verdict.supported else LABELS[1],
                        run=_rel(path),
                        audit=audit,
                    )
                )
    return claims


def extracts(audit: PerExampleAudit) -> str:
    return "\n\n".join(f"[{i + 1}] {hit.chunk.text}" for i, hit in enumerate(audit.hits_used))


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def cited_passage(claim: str, citations: Sequence[Citation]) -> str:
    """The answer's citation sharing the most words with the claim. Citations belong
    to the answer, not to a claim; the overlap picks the one the claim rests on."""
    if not citations:
        return "(the answer cited no passage)"
    words = _words(claim)
    return max(citations, key=lambda c: len(words & _words(c.quote))).quote


def load_entities(path: Path = ENTITIES) -> dict[str, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _perturb_pair(
    answer: Sequence[Claim], quota: dict[Kind, int], entities: dict[str, list[str]]
) -> list[tuple[Claim, Perturbation]] | None:
    """Two claims of one answer, each with a perturbation of a kind still needed."""
    evidence = extracts(answer[0].audit)
    for i, first in enumerate(answer):
        for k1 in sorted((k for k in Kind if quota[k]), key=lambda k: -quota[k]):
            p1 = perturb(first.claim, k1, evidence, entities)
            if p1 is None:
                continue
            left = {**quota, k1: quota[k1] - 1}
            for second in answer[i + 1 :]:
                for k2 in sorted((k for k in Kind if left[k]), key=lambda k: -left[k]):
                    p2 = perturb(second.claim, k2, evidence, entities)
                    if p2 is not None:
                        return [(first, p1), (second, p2)]
    return None


def plan(
    claims: Sequence[Claim], entities: dict[str, list[str]], seed: int = SEED
) -> tuple[list[tuple[Claim, Perturbation]], list[Claim]]:
    """``N_PER_KIND`` perturbations per kind and ``N_REAL`` real claims, exactly
    ``PER_ANSWER`` items per answer, each answer all real or all perturbed.

    Items of one answer share their extracts, so how often an extract repeats must
    not depend on origin: with every answer giving two items, it never does. Answers
    holding an unsupported claim are real, so every unsupported claim is sampled."""
    rng = random.Random(seed)
    by_answer: dict[str, list[Claim]] = {}
    for c in claims:
        by_answer.setdefault(c.answer_id, []).append(c)
    flagged = sorted({c.answer_id for c in claims if c.judge_verdict == LABELS[1]})
    if any(len(by_answer[a]) < PER_ANSWER for a in flagged):
        raise ValueError("an answer with an unsupported claim has a single claim")
    answers = sorted(a for a, cs in by_answer.items() if len(cs) >= PER_ANSWER)
    rng.shuffle(answers)
    quota = dict.fromkeys(Kind, N_PER_KIND)
    perturbed: list[tuple[Claim, Perturbation]] = []
    for answer_id in answers:
        if not any(quota.values()):
            break
        if answer_id in flagged:
            continue
        pair = _perturb_pair(by_answer[answer_id], quota, entities)
        if pair is not None:
            perturbed += pair
            for _, p in pair:
                quota[p.kind] -= 1
    short = [str(k) for k, n in quota.items() if n]
    if short:
        raise ValueError(f"not enough answers for a perturbation pair of kind {', '.join(short)}")
    real: list[Claim] = []
    for answer_id in flagged:
        bad = [c for c in by_answer[answer_id] if c.judge_verdict == LABELS[1]]
        good = [c for c in by_answer[answer_id] if c.judge_verdict == LABELS[0]]
        real += (bad + rng.sample(good, max(0, PER_ANSWER - len(bad))))[:PER_ANSWER]
    used = {c.answer_id for c, _ in perturbed} | set(flagged)
    for answer_id in answers:
        if len(real) >= N_REAL:
            break
        if answer_id not in used:
            real += rng.sample(by_answer[answer_id], PER_ANSWER)
    if len(real) != N_REAL:
        raise ValueError(f"{len(real)} real claims sampled, {N_REAL} needed")
    return perturbed, real


def export(runs: Sequence[Path]) -> int:
    claims = collect(runs)
    perturbed, real = plan(claims, load_entities())
    items: list[tuple[str, Claim, Perturbation | None]] = [
        *((p.perturbed, c, p) for c, p in perturbed),
        *((c.claim, c, None) for c in real),
    ]
    random.Random(SEED + 1).shuffle(items)
    half = len(items) // 2
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    key: dict[str, dict[str, Any]] = {}
    for index, (path, chunk) in enumerate(
        zip(CSV_FILES, (items[:half], items[half:]), strict=True)
    ):
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(COLUMNS)
            for n, (text, c, p) in enumerate(chunk, start=1):
                item_id = f"{'AB'[index]}{n:02d}"
                writer.writerow(
                    [item_id, text, cited_passage(text, c.audit.citations), extracts(c.audit), ""]
                )
                key[item_id] = {
                    "source_claim": c.claim_id,
                    "run": c.run,
                    "origin": "perturbed" if p else "real",
                    "kind": str(p.kind) if p else None,
                    "original": c.claim,
                    "changed": [p.old, p.new] if p else None,
                    "claim": text,
                    "judge_verdict": None if p else c.judge_verdict,
                }
    KEY.write_text(
        json.dumps(
            {"runs": [_rel(p) for p in runs], "seed": SEED, "items": key}, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"{len(perturbed)} perturbed + {len(real)} real claims -> {', '.join(_rel(p) for p in CSV_FILES)}"
    )
    print(f"key (do not open before annotating) -> {_rel(KEY)}")
    return 0


def _perturbed_items(key: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {k: v for k, v in key["items"].items() if v["origin"] == "perturbed"}


def _audit(run: str, claim_id: str) -> PerExampleAudit:
    example_id = claim_id.split("/")[1]
    return next(a for a in load_run(REPO_ROOT / run).per_audit if a.example_id == example_id)


def estimate() -> int:
    """Cost of ``rejudge`` from the judge calls the runs recorded: each perturbed claim
    is judged against the same evidence as its answer, with a one-claim answer."""
    key = json.loads(KEY.read_text(encoding="utf-8"))
    n = len(_perturbed_items(key))
    calls = [
        a.judge_usage
        for run in key["runs"]
        for a in load_run(REPO_ROOT / run).per_audit
        if a.judge_usage is not None and a.judge_usage.llm_calls
    ]
    if not calls:
        print("the runs recorded no judge call: nothing to estimate from", file=sys.stderr)
        return 1
    costs = sorted(u.cost_usd for u in calls)
    result = {
        "perturbed_claims": n,
        "recorded_judge_calls": len(calls),
        "per_call_usd": {
            "mean": round(statistics.mean(costs), 4),
            "p90": round(costs[int(0.9 * (len(costs) - 1))], 4),
            "max": round(costs[-1], 4),
        },
        "input_tokens_mean": round(statistics.mean(u.input_tokens for u in calls)),
        "output_tokens_mean": round(statistics.mean(u.output_tokens for u in calls)),
        "estimate_usd": {
            "expected": round(n * statistics.mean(costs), 2),
            "high": round(n * costs[-1], 2),
        },
    }
    print(json.dumps(result, indent=2))
    return 0


class Judge(Protocol):
    """The part of a ``FaithfulnessJudge`` that ``rejudge`` uses."""

    model: str
    last_usage: UsageStats | None
    last_claims: tuple[ClaimVerdict, ...] | None

    async def judge(self, example: EvalExample, answer: Answer) -> AnswerMetrics: ...


def _save(judge: Judge, spent: float, verdicts: dict[str, Any]) -> dict[str, Any]:
    result = {"judge_model": judge.model, "cost_usd": round(spent, 6), "verdicts": verdicts}
    REJUDGED.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


async def judge_perturbed(judge: Judge, budget_usd: float) -> dict[str, Any]:
    """Judge each perturbed claim as a one-claim answer, with its answer's citations
    and extracts. A claim the judge finds unsupported is detected. Stops before a call
    once ``budget_usd`` is spent; malformed judge output counts as no verdict.

    Each verdict is written as soon as it is paid for, and a rerun skips the items
    already in the file: a provider error stops the loop without losing them, and
    ``budget_usd`` bounds this invocation's spend."""
    from verity.llm.base import LLMError

    key = json.loads(KEY.read_text(encoding="utf-8"))
    previous = json.loads(REJUDGED.read_text(encoding="utf-8")) if REJUDGED.is_file() else {}
    verdicts: dict[str, Any] = dict(previous.get("verdicts", {}))
    total = float(previous.get("cost_usd", 0.0))
    spent = 0.0
    result = {"judge_model": judge.model, "cost_usd": total, "verdicts": verdicts}
    for item_id, item in sorted(_perturbed_items(key).items()):
        if item_id in verdicts:
            continue
        if spent >= budget_usd:
            print(
                f"budget ${budget_usd:.2f} spent: {item_id} and later not judged", file=sys.stderr
            )
            break
        audit = _audit(item["run"], item["source_claim"])
        example = EvalExample(
            id=audit.example_id, question=audit.question, answerable=audit.expected_answerable
        )
        answer = Answer(
            query=audit.question,
            text=item["claim"],
            citations=audit.citations,
            hits_used=audit.hits_used,
            # The judge reads text, citations and extracts; the score is not shown to it.
            confidence=Confidence(
                score=audit.confidence_score or 0.0,
                refused=False,
                rationale="perturbed claim for judge validation",
            ),
        )
        try:
            await judge.judge(example, answer)
        except LLMError as exc:
            # Typed provider failure: keep what is paid for, stop, let a rerun resume.
            print(f"{item_id}: {type(exc).__name__}: {exc}; rerun to resume", file=sys.stderr)
            break
        cost = judge.last_usage.cost_usd if judge.last_usage else 0.0
        spent += cost
        total += cost
        claims = judge.last_claims or ()
        detected = any(not v.supported for v in claims)
        verdicts[item_id] = {
            "verdict": (LABELS[1] if detected else LABELS[0]) if claims else None,
            "claims": [{"claim": v.claim, "supported": v.supported} for v in claims],
            "cost_usd": cost,
        }
        result = _save(judge, total, verdicts)
    return result


def rejudge(budget_usd: float) -> int:
    from verity.eval.judge import LLMFaithfulnessJudge
    from verity.llm import create_default_routing_client
    from verity.llm.routing import create_judge_client

    # The agent client is built only so the judge factory can refuse a judge that is
    # the agent's own model; it makes no call.
    judge = LLMFaithfulnessJudge(create_judge_client(agent_client=create_default_routing_client()))
    result = asyncio.run(judge_perturbed(judge, budget_usd))
    print(f"{len(result['verdicts'])} perturbed claims judged, ${result['cost_usd']:.4f}")
    return 0


def cohen_kappa(pairs: Sequence[tuple[str, str]]) -> float | None:
    """κ = (p_o - p_e) / (1 - p_e) over (judge, human) label pairs; ``None`` when
    undefined (no pairs, or both raters used a single identical label throughout)."""
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(1 for a, b in pairs if a == b) / n
    expected = sum(
        (sum(1 for a, _ in pairs if a == label) / n) * (sum(1 for _, b in pairs if b == label) / n)
        for label in LABELS
    )
    if expected == 1.0:
        return None
    return (observed - expected) / (1 - expected)


def _proportion(k: int, n: int) -> dict[str, Any]:
    ci = wilson(k, n)
    return {
        "k": k,
        "n": n,
        "value": round(k / n, 6) if n else None,
        "ci95_wilson": None if ci is None else [round(ci[0], 4), round(ci[1], 4)],
    }


def _read_annotations() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in CSV_FILES:
        with path.open(newline="", encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def _pairs_by_id(
    judge: dict[str, str | None], human: dict[str, str], key: dict[str, Any]
) -> list[tuple[str, str] | None]:
    return [(j, human[i]) if (j := judge[i]) is not None else None for i in sorted(key)]


def _agreement(pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
    return {
        "n_claims": len(pairs),
        "observed": _proportion(sum(1 for a, b in pairs if a == b), len(pairs)),
        "cohen_kappa": cohen_kappa(pairs),
        "confusion": {
            f"judge_{a}/human_{b}": sum(1 for p in pairs if p == (a, b))
            for a in LABELS
            for b in LABELS
        },
    }


def _inputs() -> tuple[dict[str, Any], dict[str, str | None], list[dict[str, str]]] | str:
    """The key, the judge's verdict per item and the blind rows, or why they are unusable."""
    key = json.loads(KEY.read_text(encoding="utf-8"))["items"]
    if not REJUDGED.is_file():
        return f"{_rel(REJUDGED)} missing: run `rejudge` first"
    rejudged = json.loads(REJUDGED.read_text(encoding="utf-8"))["verdicts"]
    rows = _read_annotations()
    unknown = [r["item_id"] for r in rows if r["item_id"] not in key]
    ids = [r["item_id"] for r in rows]
    if unknown or len(ids) != len(set(ids)) or set(ids) != set(key):
        return (
            f"the CSVs and {_rel(KEY)} do not match: "
            f"{', '.join(unknown) or 'duplicate or missing ids'}"
        )
    bad = [r["item_id"] for r in rows if r["human_verdict"].strip().lower() not in LABELS]
    if bad:
        return f"human_verdict must be one of {LABELS}; fix: {', '.join(bad)}"
    judge: dict[str, str | None] = {
        i: (
            rejudged.get(i, {}).get("verdict")
            if item["origin"] == "perturbed"
            else item["judge_verdict"]
        )
        for i, item in key.items()
    }
    return key, judge, rows


def measure(
    key: dict[str, Any], judge: dict[str, str | None], human: dict[str, str]
) -> dict[str, Any]:
    """Agreement, specificity and sensitivity of ``judge`` against ``human`` labels."""
    missing = sorted(i for i, v in judge.items() if v is None)
    pairs = [(j, human[i]) for i in sorted(key) if (j := judge[i]) is not None]
    real_pairs = [
        p
        for i, p in zip(sorted(key), _pairs_by_id(judge, human, key), strict=True)
        if p is not None and key[i]["origin"] == "real"
    ]
    perturbed = sorted(i for i, item in key.items() if item["origin"] == "perturbed")
    judged = [i for i in perturbed if judge[i] is not None]
    by_kind: dict[str, Any] = {}
    for kind in Kind:
        ids = [i for i in judged if key[i]["kind"] == kind]
        by_kind[kind] = _proportion(sum(1 for i in ids if judge[i] == LABELS[1]), len(ids))
    return {
        # (1) how many false claims the judge catches, on its own and among those
        # the human also found unsupported (a perturbation can leave a claim true).
        "sensitivity": {
            "judge_detected": _proportion(
                sum(1 for i in judged if judge[i] == LABELS[1]), len(judged)
            ),
            "human_unsupported": _proportion(
                sum(1 for i in perturbed if human[i] == LABELS[1]), len(perturbed)
            ),
            "judge_detected_among_human_unsupported": _proportion(
                sum(1 for i in judged if human[i] == LABELS[1] and judge[i] == LABELS[1]),
                sum(1 for i in judged if human[i] == LABELS[1]),
            ),
            "by_kind": by_kind,
        },
        # (2) agreement on the mixed set, the human blind to origin and verdict. Real
        # items carry the live verdict (the judge split the whole answer), perturbed
        # ones the one-claim verdict: two protocols, so the real-only figures follow.
        "agreement_mixed": _agreement(pairs),
        "agreement_real_only": _agreement(real_pairs),
        "specificity_real": _proportion(
            sum(1 for j, h in real_pairs if h == LABELS[0] and j == LABELS[0]),
            sum(1 for _, h in real_pairs if h == LABELS[0]),
        ),
        "notes": [
            "kappa has no Wilson interval: it is not a proportion; with 60 items it is "
            "uncertain by itself",
            "a perturbed claim counts as detected if any sub-claim the judge made of it is "
            "unsupported, even outside the changed span: sensitivity is an upper bound "
            "(per-claim detail in perturbed_verdicts.json)",
        ],
        "judge_without_verdict": missing,
    }


def kappa() -> int:
    inputs = _inputs()
    if isinstance(inputs, str):
        print(inputs, file=sys.stderr)
        return 1
    key, judge, rows = inputs
    human = {r["item_id"]: r["human_verdict"].strip().lower() for r in rows}
    result = measure(key, judge, human)
    with FINAL_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["item_id", "origin", "kind", "claim", "original", "judge_verdict", "human_verdict"]
        )
        for r in sorted(rows, key=lambda r: r["item_id"]):
            item = key[r["item_id"]]
            writer.writerow(
                [
                    r["item_id"],
                    item["origin"],
                    item["kind"] or "",
                    r["claim"],
                    item["original"],
                    judge[r["item_id"]] or "",
                    human[r["item_id"]],
                ]
            )
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _to_adjudicate(
    key: dict[str, Any], judge: dict[str, str | None], human: dict[str, str]
) -> list[str]:
    """Every disagreement, plus every perturbed claim the annotator judged true: those
    are categorised after adjudication, and some of them the judge accepted too."""
    disagree = {i for i in key if judge[i] is not None and judge[i] != human[i]}
    perturbed_true = {
        i for i, item in key.items() if item["origin"] == "perturbed" and human[i] == LABELS[0]
    }
    return sorted(disagree | perturbed_true)


def adjudication_export() -> int:
    if ADJUDICATION.exists():
        print(
            f"{_rel(ADJUDICATION)} exists: it may hold decisions; not overwritten", file=sys.stderr
        )
        return 1
    inputs = _inputs()
    if isinstance(inputs, str):
        print(inputs, file=sys.stderr)
        return 1
    key, judge, rows = inputs
    human = {r["item_id"]: r["human_verdict"].strip().lower() for r in rows}
    by_id = {r["item_id"]: r for r in rows}
    ids = _to_adjudicate(key, judge, human)
    with ADJUDICATION.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(ADJ_COLUMNS)
        for i in ids:
            r = by_id[i]
            writer.writerow(
                [i, r["claim"], r["cited_passage"], r["extracts"], human[i], judge[i] or "", "", ""]
            )
    print(f"{len(ids)} claims to adjudicate in {_rel(ADJUDICATION)}")
    return 0


def adjudicated() -> int:
    """Measure the judge against the adjudicated labels. Writes only ADJUDICATED: the
    blind labels and agreement.json stay the published result of record."""
    inputs = _inputs()
    if isinstance(inputs, str):
        print(inputs, file=sys.stderr)
        return 1
    key, judge, rows = inputs
    blind = {r["item_id"]: r["human_verdict"].strip().lower() for r in rows}
    with ADJUDICATION.open(newline="", encoding="utf-8") as fh:
        decisions = list(csv.DictReader(fh))
    expected = _to_adjudicate(key, judge, blind)
    if sorted(d["item_id"] for d in decisions) != expected:
        print(f"{_rel(ADJUDICATION)} does not list the claims to adjudicate", file=sys.stderr)
        return 1
    bad = [
        d["item_id"]
        for d in decisions
        if d["adjudicated_verdict"].strip().lower() not in LABELS
        or not d["reason"].strip()
        or d["blind_verdict"] != blind[d["item_id"]]
    ]
    if bad:
        print(
            f"verdict, reason or blind label missing or altered: {', '.join(bad)}", file=sys.stderr
        )
        return 1
    final = dict(blind)
    log = []
    for d in sorted(decisions, key=lambda d: d["item_id"]):
        i, verdict = d["item_id"], d["adjudicated_verdict"].strip().lower()
        final[i] = verdict
        item = key[i]
        entry = {
            "item_id": i,
            "origin": item["origin"],
            "kind": item["kind"],
            "blind": blind[i],
            "judge": judge[i],
            "adjudicated": verdict,
            "changed": verdict != blind[i],
            "reason": d["reason"].strip(),
        }
        if item["origin"] == "perturbed" and blind[i] == LABELS[0]:
            entry["category"] = STAYS_TRUE if verdict == LABELS[0] else ANNOTATOR_ERROR
        log.append(entry)
    result = measure(key, judge, final)
    categories = [e["category"] for e in log if "category" in e]
    result["adjudication"] = {
        "n_adjudicated": len(log),
        "n_changed": sum(1 for e in log if e["changed"]),
        # The judge's verdict was visible during adjudication: a change that lands on it
        # may be persuasion as much as correction, so the count is published.
        "n_changed_to_judge": sum(
            1 for e in log if e["changed"] and e["adjudicated"] == e["judge"]
        ),
        "perturbed_judged_true_blind": {
            "n": len(categories),
            STAYS_TRUE: categories.count(STAYS_TRUE),
            ANNOTATOR_ERROR: categories.count(ANNOTATOR_ERROR),
        },
        "items": log,
        "note": "adjudicated with the judge's verdict visible, by the annotator of the "
        "blind labels; the blind result (agreement.json) is the result of record",
    }
    ADJUDICATED.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["adjudication"], indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("export")
    ex.add_argument("runs", type=Path, nargs="+")
    sub.add_parser("estimate")
    rj = sub.add_parser("rejudge")
    rj.add_argument("--budget-usd", type=float, required=True)
    sub.add_parser("kappa")
    sub.add_parser("adjudication-export")
    sub.add_parser("adjudicated")
    args = parser.parse_args(argv)
    if args.command == "export":
        return export(args.runs)
    if args.command == "estimate":
        return estimate()
    if args.command == "rejudge":
        return rejudge(args.budget_usd)
    if args.command == "adjudication-export":
        return adjudication_export()
    if args.command == "adjudicated":
        return adjudicated()
    return kappa()


if __name__ == "__main__":
    raise SystemExit(main())
