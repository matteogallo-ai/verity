"""Judge validation: export judged claims for human annotation, then compute Cohen's κ.

    # 1. sample claims from the live dev and test runs; writes the blind CSV + the key
    uv run python scripts/judge_agreement.py export RUN.json [RUN.json ...]
    # 2. a human fills `human_verdict` (supported / unsupported) in the blind CSV
    # 3. join with the judge's verdicts, compute κ, write the final CSV and the result
    uv run python scripts/judge_agreement.py kappa

The annotator sees the claim and the context the judge saw (the retrieved chunks), not
the judge's verdict: knowing it would anchor the human label and inflate agreement.
The verdicts stay in a key file until ``kappa`` joins them into the final CSV, which
then has the columns ``claim, evidence, judge_verdict, human_verdict``.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from verity.eval.run_store import load_run

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "datasets" / "eval" / "judge_validation"
BLIND_CSV = OUT_DIR / "claims_to_annotate.csv"
KEY = OUT_DIR / "judge_verdicts.json"
FINAL_CSV = OUT_DIR / "claims_annotated.csv"
RESULT = OUT_DIR / "agreement.json"

SEED = 20261102
N_CLAIMS = 60
LABELS = ("supported", "unsupported")


@dataclass(frozen=True)
class Claim:
    claim_id: str
    claim: str
    evidence: str
    judge_verdict: str


def collect(runs: Sequence[Path]) -> list[Claim]:
    """Every claim the judge returned in ``runs``, with the context it judged against."""
    claims: list[Claim] = []
    for path in runs:
        run = load_run(path)
        for audit in run.per_audit:
            evidence = "\n\n".join(
                f"[{i + 1}] {hit.chunk.text}" for i, hit in enumerate(audit.hits_used)
            )
            for j, verdict in enumerate(audit.judge_claims or ()):
                claims.append(
                    Claim(
                        claim_id=f"{run.scorecard.dataset}/{audit.example_id}/{j}",
                        claim=verdict.claim,
                        evidence=evidence,
                        judge_verdict=LABELS[0] if verdict.supported else LABELS[1],
                    )
                )
    return claims


def sample(claims: Sequence[Claim], n: int = N_CLAIMS, seed: int = SEED) -> list[Claim]:
    """A uniform random sample (all claims when there are at most ``n``), in id order."""
    if len(claims) <= n:
        return sorted(claims, key=lambda c: c.claim_id)
    return sorted(random.Random(seed).sample(list(claims), n), key=lambda c: c.claim_id)


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


def export(runs: Sequence[Path]) -> int:
    chosen = sample(collect(runs))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with BLIND_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["claim_id", "claim", "evidence", "human_verdict"])
        for c in chosen:
            writer.writerow([c.claim_id, c.claim, c.evidence, ""])
    KEY.write_text(
        json.dumps(
            {
                "runs": [p.resolve().relative_to(REPO_ROOT).as_posix() for p in runs],
                "seed": SEED,
                "verdicts": {c.claim_id: c.judge_verdict for c in chosen},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"{len(chosen)} claims -> {BLIND_CSV}; judge verdicts -> {KEY}")
    return 0


def kappa() -> int:
    key = json.loads(KEY.read_text(encoding="utf-8"))["verdicts"]
    with BLIND_CSV.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    bad = [r["claim_id"] for r in rows if r["human_verdict"].strip().lower() not in LABELS]
    if bad:
        print(f"human_verdict must be one of {LABELS}; fix: {', '.join(bad)}", file=sys.stderr)
        return 1
    pairs = [(key[r["claim_id"]], r["human_verdict"].strip().lower()) for r in rows]
    with FINAL_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["claim_id", "claim", "evidence", "judge_verdict", "human_verdict"])
        for r, (judge, human) in zip(rows, pairs, strict=True):
            writer.writerow([r["claim_id"], r["claim"], r["evidence"], judge, human])
    result = {
        "n_claims": len(pairs),
        "agreement": sum(1 for a, b in pairs if a == b) / len(pairs),
        "cohen_kappa": cohen_kappa(pairs),
        "confusion": {
            f"judge_{a}/human_{b}": sum(1 for p in pairs if p == (a, b))
            for a in LABELS
            for b in LABELS
        },
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("export")
    ex.add_argument("runs", type=Path, nargs="+")
    sub.add_parser("kappa")
    args = parser.parse_args(argv)
    return export(args.runs) if args.command == "export" else kappa()


if __name__ == "__main__":
    raise SystemExit(main())
