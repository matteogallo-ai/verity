"""Recompute each live run's answer-quality aggregates with the current definition.

A published run artefact is immutable, even its derived aggregates. When the metric
definition changes (``verity.eval.answer_quality``), the recomputed figures go into a
sibling file ``<artefact>.recomputed.json`` next to the untouched artefact, with the
stored and the recomputed values side by side.

    uv run python scripts/recompute_live_aggregates.py           # write the files
    uv run python scripts/recompute_live_aggregates.py --check   # exit 1 if stale
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from verity.eval.answer_quality import Ratio, answer_quality
from verity.eval.run_store import load_run

LIVE_DIR = Path(__file__).resolve().parents[1] / "datasets" / "eval" / "runs" / "live"
SUFFIX = ".recomputed.json"


def artefacts(live_dir: Path = LIVE_DIR) -> list[Path]:
    return sorted(p for p in live_dir.glob("eval_*.json") if not p.name.endswith(SUFFIX))


def recomputed_path(artefact: Path) -> Path:
    return artefact.with_name(artefact.stem + SUFFIX)


def _ratio(r: Ratio) -> dict[str, Any]:
    return {"value": r.value, "numerator": r.numerator, "denominator": r.denominator}


def recompute(artefact: Path) -> dict[str, Any]:
    run = load_run(artefact)
    q = answer_quality(run.per_audit)
    stored = run.scorecard.answer
    return {
        "source": artefact.name,
        "source_sha256": hashlib.sha256(artefact.read_bytes()).hexdigest(),
        "definition": "verity.eval.answer_quality (docs/evaluation-methodology.md)",
        "stored": {
            "faithfulness": stored.faithfulness,
            "citation_accuracy": stored.citation_accuracy,
            "hallucination_rate": stored.hallucination_rate,
        },
        "recomputed": {
            "n_judged": q.n_judged,
            "n_judged_without_claims": q.n_judged_without_claims,
            "faithfulness_macro": _ratio(q.faithfulness_macro),
            "faithfulness_micro": _ratio(q.faithfulness_micro),
            "citation_accuracy_macro": _ratio(q.citation_accuracy_macro),
            "citation_accuracy_micro": _ratio(q.citation_accuracy_micro),
            "hallucination_rate": _ratio(q.hallucination_rate),
        },
    }


def render(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if a file is stale")
    args = parser.parse_args(argv)
    stale: list[str] = []
    for artefact in artefacts():
        target = recomputed_path(artefact)
        text = render(recompute(artefact))
        if args.check:
            if not target.is_file() or target.read_text(encoding="utf-8") != text:
                stale.append(target.name)
        else:
            target.write_text(text, encoding="utf-8")
            print(f"wrote {target.name}")
    if stale:
        print(f"stale: {', '.join(stale)} — run without --check", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
