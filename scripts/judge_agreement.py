"""Judge validation: blind annotation of real and perturbed claims.

    # 1. sample 40 real claims (every unsupported one included) and make 20 false
    #    ones (verity.eval.perturb); writes two blind CSVs of 30 and the key ($0)
    uv run python scripts/judge_agreement.py export RUN.json [RUN.json ...]
    # 2. a human fills `human_verdict` (supported / unsupported) in both CSVs

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
import csv
import json
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity.eval.perturb import Kind, Perturbation, perturb
from verity.eval.run_store import load_run
from verity.types import (
    Citation,
    PerExampleAudit,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "datasets" / "eval" / "judge_validation"
ENTITIES = REPO_ROOT / "datasets" / "eval" / "entities.json"
CSV_FILES = (OUT_DIR / "annotate_morning_1.csv", OUT_DIR / "annotate_morning_2.csv")
KEY = OUT_DIR / "judge_key.json"

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("export")
    ex.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args(argv)
    return export(args.runs)


if __name__ == "__main__":
    raise SystemExit(main())
