"""Question files for completing the naive-baseline comparison on dev.

Both arms of docs/preregistration/naive-rag-baseline.md stopped on their budget. The
amendment of 2026-11-05 runs each arm on the dev questions it did not complete, and
nothing else: no completed question is run again.

- ``dev-completion.jsonl``: the questions the Verity arm did not complete.
- ``dev-completion-11.jsonl``: the same rows minus those the naive arm completed.

Rows are copied unchanged from ``dev.jsonl``. Their order is ``random.Random(SEED)``
applied to the sorted ids, and the naive file keeps that order: when a budget stops a
run, the questions left out are a random draw, not the end of the id order (where
dev keeps most of its unanswerable questions).

    uv run python scripts/build_completion_subset.py           # write both files
    uv run python scripts/build_completion_subset.py --check   # exit 1 if stale
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

SEED = 20261105

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "datasets" / "eval"
LIVE = EVAL_DIR / "runs" / "live"
DEV = EVAL_DIR / "dev.jsonl"
VERITY_RUN = LIVE / "eval_e644a98_dev.incomplete-20261105T094431636600Z.json"
NAIVE_RUN = LIVE / "eval_e644a98_dev-naive-baseline.incomplete-20261105T094823388125Z.json"
VERITY_OUT = EVAL_DIR / "dev-completion.jsonl"
NAIVE_OUT = EVAL_DIR / "dev-completion-11.jsonl"


def completed(run: Path) -> set[str]:
    """Ids the run answered or refused; errored questions are not completed."""
    return {row["example_id"] for row in json.loads(run.read_text())["per_audit"]}


def subsets(
    dev: list[dict], verity_done: set[str], naive_done: set[str], seed: int = SEED
) -> tuple[list[dict], list[dict]]:  # type: ignore[type-arg]
    by_id = {row["id"]: row for row in dev}
    missing = sorted(set(by_id) - verity_done)
    random.Random(seed).shuffle(missing)
    verity = [by_id[i] for i in missing]
    naive = [row for row in verity if row["id"] not in naive_done]
    return verity, naive


def _render(rows: list[dict]) -> str:  # type: ignore[type-arg]
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the files are stale")
    args = parser.parse_args(argv)
    dev = [json.loads(line) for line in DEV.read_text().splitlines() if line.strip()]
    verity, naive = subsets(dev, completed(VERITY_RUN), completed(NAIVE_RUN))
    expected = {VERITY_OUT: _render(verity), NAIVE_OUT: _render(naive)}
    if args.check:
        stale = [p.name for p, text in expected.items() if not p.exists() or p.read_text() != text]
        if stale:
            print(f"stale: {', '.join(stale)}; run scripts/build_completion_subset.py")
            return 1
        print(f"{VERITY_OUT.name} ({len(verity)}) and {NAIVE_OUT.name} ({len(naive)}) current")
        return 0
    for path, text in expected.items():
        path.write_text(text)
    print(f"wrote {VERITY_OUT.name} ({len(verity)}) and {NAIVE_OUT.name} ({len(naive)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
