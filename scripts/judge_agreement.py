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

Why perturbed claims: the judge marked 161 of the 164 live claims supported, so on
real claims alone κ is undefined or 0 whatever the judge does. A perturbed claim is
the original with one span changed (figure, date, negation, entity of the same type).

Blindness: the CSVs hold an opaque item id, the claim, the passage its answer cited
(the citation closest to the claim, first), then every extract the judge saw. They
do not say which claims are perturbed, nor any verdict; ``judge_key.json`` holds
both and must stay closed until the annotation is done. A perturbed claim and real
claims from the same answer are never both in the sample: side by side, the
disagreement would give the perturbation away.
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

SEED = 20261102
N_REAL = 40
N_PER_KIND = 5
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


def plan(
    claims: Sequence[Claim], entities: dict[str, list[str]], seed: int = SEED
) -> tuple[list[tuple[Claim, Perturbation]], list[Claim]]:
    """``N_PER_KIND`` perturbations per kind from distinct answers with no unsupported
    claim, then ``N_REAL`` real claims from the other answers, every claim the judge
    marked unsupported among them."""
    rng = random.Random(seed)
    supported = [c for c in claims if c.judge_verdict == LABELS[0]]
    rng.shuffle(supported)
    perturbed: list[tuple[Claim, Perturbation]] = []
    # Answers holding an unsupported claim are never perturbed: their unsupported
    # claims all go into the real sample, which a perturbation would push out.
    used: set[str] = {c.answer_id for c in claims if c.judge_verdict == LABELS[1]}
    for kind in Kind:
        picked = 0
        for c in supported:
            if picked == N_PER_KIND:
                break
            if c.answer_id in used:
                continue
            p = perturb(c.claim, kind, extracts(c.audit), entities)
            if p is not None:
                perturbed.append((c, p))
                used.add(c.answer_id)
                picked += 1
        if picked < N_PER_KIND:
            raise ValueError(f"only {picked} claims can take a {kind} perturbation")
    perturbed_answers = {c.answer_id for c, _ in perturbed}
    pool = [c for c in claims if c.answer_id not in perturbed_answers]
    unsupported = [c for c in pool if c.judge_verdict == LABELS[1]]
    rest = [c for c in pool if c.judge_verdict == LABELS[0]]
    real = unsupported + rng.sample(rest, N_REAL - len(unsupported))
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


async def judge_perturbed(judge: Judge, budget_usd: float) -> dict[str, Any]:
    """Judge each perturbed claim as a one-claim answer, with its answer's citations
    and extracts. A claim the judge finds unsupported is detected. Stops before a call
    once ``budget_usd`` is spent; malformed judge output counts as no verdict."""
    key = json.loads(KEY.read_text(encoding="utf-8"))
    verdicts: dict[str, Any] = {}
    spent = 0.0
    for item_id, item in sorted(_perturbed_items(key).items()):
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
        await judge.judge(example, answer)
        cost = judge.last_usage.cost_usd if judge.last_usage else 0.0
        spent += cost
        claims = judge.last_claims or ()
        detected = any(not v.supported for v in claims)
        verdicts[item_id] = {
            "verdict": (LABELS[1] if detected else LABELS[0]) if claims else None,
            "claims": [{"claim": v.claim, "supported": v.supported} for v in claims],
            "cost_usd": cost,
        }
    return {"judge_model": judge.model, "cost_usd": round(spent, 6), "verdicts": verdicts}


def rejudge(budget_usd: float) -> int:
    from verity.eval.judge import LLMFaithfulnessJudge
    from verity.llm import create_default_routing_client
    from verity.llm.routing import create_judge_client

    # The agent client is built only so the judge factory can refuse a judge that is
    # the agent's own model; it makes no call.
    judge = LLMFaithfulnessJudge(create_judge_client(agent_client=create_default_routing_client()))
    result = asyncio.run(judge_perturbed(judge, budget_usd))
    REJUDGED.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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


def kappa() -> int:
    key = json.loads(KEY.read_text(encoding="utf-8"))["items"]
    if not REJUDGED.is_file():
        print(f"{_rel(REJUDGED)} missing: run `rejudge` first", file=sys.stderr)
        return 1
    rejudged = json.loads(REJUDGED.read_text(encoding="utf-8"))["verdicts"]
    rows = _read_annotations()
    unknown = [r["item_id"] for r in rows if r["item_id"] not in key]
    if unknown or len(rows) != len(key):
        print(
            f"the CSVs and {_rel(KEY)} do not match: {', '.join(unknown) or 'count'}",
            file=sys.stderr,
        )
        return 1
    bad = [r["item_id"] for r in rows if r["human_verdict"].strip().lower() not in LABELS]
    if bad:
        print(f"human_verdict must be one of {LABELS}; fix: {', '.join(bad)}", file=sys.stderr)
        return 1
    human = {r["item_id"]: r["human_verdict"].strip().lower() for r in rows}
    judge: dict[str, str | None] = {
        i: (
            rejudged.get(i, {}).get("verdict")
            if item["origin"] == "perturbed"
            else item["judge_verdict"]
        )
        for i, item in key.items()
    }
    missing = sorted(i for i, v in judge.items() if v is None)
    pairs = [(j, human[i]) for i in sorted(key) if (j := judge[i]) is not None]
    perturbed = sorted(i for i, item in key.items() if item["origin"] == "perturbed")
    judged = [i for i in perturbed if judge[i] is not None]
    by_kind: dict[str, Any] = {}
    for kind in Kind:
        ids = [i for i in judged if key[i]["kind"] == kind]
        by_kind[kind] = _proportion(sum(1 for i in ids if judge[i] == LABELS[1]), len(ids))
    result = {
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
        # (2) agreement on the mixed set, the human blind to origin and verdict.
        "agreement_mixed": {
            "n_claims": len(pairs),
            "observed": _proportion(sum(1 for a, b in pairs if a == b), len(pairs)),
            "cohen_kappa": cohen_kappa(pairs),
            "confusion": {
                f"judge_{a}/human_{b}": sum(1 for p in pairs if p == (a, b))
                for a in LABELS
                for b in LABELS
            },
        },
        "judge_without_verdict": missing,
    }
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
    args = parser.parse_args(argv)
    if args.command == "export":
        return export(args.runs)
    if args.command == "estimate":
        return estimate()
    if args.command == "rejudge":
        return rejudge(args.budget_usd)
    return kappa()


if __name__ == "__main__":
    raise SystemExit(main())
