"""Split the eval dataset into a frozen dev set and test set.

The refusal threshold (and any calibration) is tuned on ``dev.jsonl`` only; published
results are measured on ``test.jsonl``. The split is stratified so both halves carry
answerable, unanswerable and adversarial questions in the same proportions:
within each stratum the question ids are sorted, shuffled with ``random.Random(SEED)``
and the first ``DEV_FRACTION`` go to dev.

Changing ``SEED`` or ``DEV_FRACTION`` re-draws the split and invalidates every result
tuned on the old dev set; ``--check`` (run in CI) fails when the committed files differ
from what this script produces.

    uv run python scripts/split_dataset.py           # write dev.jsonl and test.jsonl
    uv run python scripts/split_dataset.py --check   # exit 1 if they are stale
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

SEED = 20261102
DEV_FRACTION = 0.4

EVAL_DIR = Path(__file__).resolve().parents[1] / "datasets" / "eval"
SOURCE = EVAL_DIR / "questions.jsonl"
DEV = EVAL_DIR / "dev.jsonl"
TEST = EVAL_DIR / "test.jsonl"


def _stratum(record: dict) -> str:  # type: ignore[type-arg]
    if "adversarial" in record["tags"]:
        return "adversarial"
    return "answerable" if record["answerable"] else "unanswerable"


def split(records: list[dict]) -> tuple[list[dict], list[dict]]:  # type: ignore[type-arg]
    """``(dev, test)``, each in the source order."""
    dev_ids: set[str] = set()
    rng = random.Random(SEED)
    for stratum in ("answerable", "unanswerable", "adversarial"):
        ids = sorted(r["id"] for r in records if _stratum(r) == stratum)
        rng.shuffle(ids)
        dev_ids.update(ids[: round(len(ids) * DEV_FRACTION)])
    dev = [r for r in records if r["id"] in dev_ids]
    test = [r for r in records if r["id"] not in dev_ids]
    return dev, test


def _render(records: list[dict]) -> str:  # type: ignore[type-arg]
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if dev/test are stale")
    args = parser.parse_args(argv)
    records = [json.loads(line) for line in SOURCE.read_text().splitlines() if line.strip()]
    dev, test = split(records)
    outputs = {DEV: _render(dev), TEST: _render(test)}
    if args.check:
        stale = [p.name for p, text in outputs.items() if not p.exists() or p.read_text() != text]
        if stale:
            print(f"stale split: {', '.join(stale)}", file=sys.stderr)
            return 1
        print(f"split up to date: dev {len(dev)}, test {len(test)} (seed {SEED})")
        return 0
    for path, text in outputs.items():
        path.write_text(text)
    print(f"dev {len(dev)}, test {len(test)} (seed {SEED})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
