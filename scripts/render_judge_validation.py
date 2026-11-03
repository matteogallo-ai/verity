"""Write the README's judge-validation section from the committed agreement artefact.

    uv run python scripts/render_judge_validation.py          # rewrite the block
    uv run python scripts/render_judge_validation.py --check  # CI: README == artefact

The block sits between ``<!-- JUDGE:START -->`` and ``<!-- JUDGE:END -->`` and is
rendered from ``datasets/eval/judge_validation/agreement.json``: the blind human
labels are the reference, and nothing in the block is typed by hand.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATION = REPO_ROOT / "datasets" / "eval" / "judge_validation"
AGREEMENT = VALIDATION / "agreement.json"
REJUDGED = VALIDATION / "perturbed_verdicts.json"
README = REPO_ROOT / "README.md"
START, END = "<!-- JUDGE:START -->", "<!-- JUDGE:END -->"
# Below this many items in the smaller human class, one or two disagreements move κ
# far enough that its value says little: the figure is published as fragile.
FRAGILE_BELOW = 5


def _ci(p: dict[str, Any]) -> str:
    ci = p["ci95_wilson"]
    interval = "no interval" if ci is None else f"95 % CI [{ci[0]:.3f}, {ci[1]:.3f}]"
    return f"{p['k']}/{p['n']} · {interval}"


def _value(p: dict[str, Any]) -> str:
    return "n/a" if p["value"] is None else f"{p['value']:.3f}"


def _human_counts(agreement: dict[str, Any]) -> tuple[int, int]:
    c = agreement["confusion"]
    supported = c["judge_supported/human_supported"] + c["judge_unsupported/human_supported"]
    unsupported = c["judge_supported/human_unsupported"] + c["judge_unsupported/human_unsupported"]
    return supported, unsupported


def _kappa_row(label: str, agreement: dict[str, Any]) -> str:
    supported, unsupported = _human_counts(agreement)
    fragile = min(supported, unsupported) < FRAGILE_BELOW
    small = min(supported, unsupported)
    note = (
        f"**fragile**: only {small} of {agreement['n_claims']} claims "
        f"{'is' if small == 1 else 'are'} in the smaller human class, so one or two "
        "disagreements move κ widely"
        if fragile
        else f"{supported} supported / {unsupported} unsupported by the human"
    )
    kappa = agreement["cohen_kappa"]
    shown = "undefined" if kappa is None else f"{kappa:.2f}"
    return (
        f"| {label} | {shown} | {agreement['n_claims']} claims · observed agreement "
        f"{_value(agreement['observed'])} ({_ci(agreement['observed'])}) · {note} |"
    )


def render_block(result: dict[str, Any], judge_model: str) -> str:
    s = result["sensitivity"]
    by_kind = " · ".join(f"{k} {v['k']}/{v['n']}" for k, v in sorted(s["by_kind"].items()))
    mixed, real = result["agreement_mixed"], result["agreement_real_only"]
    c = mixed["confusion"]
    lines = [
        f"Judge `{judge_model}` against **blind human labels**, the reference: "
        f"{mixed['n_claims']} claims labelled without origin or judge verdict, "
        f"{real['n_claims']} real claims from the test run and "
        f"{s['judge_detected']['n']} claims perturbed to be false "
        "(one number, date, negation or entity changed).",
        "",
        "| Measure | Value | n · notes |",
        "|---|---|---|",
        _kappa_row("Cohen's κ, all claims", mixed),
        _kappa_row("Cohen's κ, real claims only", real),
        f"| Judge specificity on real claims (human: supported) | "
        f"{_value(result['specificity_real'])} | {_ci(result['specificity_real'])} |",
        f"| Sensitivity: perturbed claims the judge finds unsupported "
        f"(**upper bound**) | {_value(s['judge_detected'])} | {_ci(s['judge_detected'])} · "
        f"by kind: {by_kind} |",
        f"| … among perturbed claims the human finds unsupported | "
        f"{_value(s['judge_detected_among_human_unsupported'])} | "
        f"{_ci(s['judge_detected_among_human_unsupported'])} |",
        f"| Perturbed claims the human finds unsupported | "
        f"{_value(s['human_unsupported'])} | {_ci(s['human_unsupported'])} |",
        "",
        f"Disagreements: the judge rejects {c['judge_unsupported/human_supported']} claims the "
        f"human accepts and accepts {c['judge_supported/human_unsupported']} the human "
        "rejects. Notes from the artefact:",
        "",
        *(f"- {note}." for note in result["notes"]),
        "",
        "Generated from "
        "[`datasets/eval/judge_validation/agreement.json`](datasets/eval/judge_validation/agreement.json) "
        "(`scripts/render_judge_validation.py --check`); the labels and the key are pinned in "
        "`datasets/eval/judge_validation/SHA256SUMS`.",
    ]
    return "\n".join(lines)


def replace_block(text: str, block: str, start: str = START, end: str = END) -> str:
    i, j = text.find(start), text.find(end)
    if i == -1 or j == -1 or j < i:
        raise SystemExit(f"README missing {start!r} / {end!r} markers")
    return f"{text[: i + len(start)]}\n{block}\n{text[j:]}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", type=Path, default=README)
    parser.add_argument("--check", action="store_true", help="exit 1 if the README would change")
    args = parser.parse_args(argv)
    result = json.loads(AGREEMENT.read_text(encoding="utf-8"))
    judge_model = json.loads(REJUDGED.read_text(encoding="utf-8"))["judge_model"]
    current = args.readme.read_text(encoding="utf-8")
    rendered = replace_block(current, render_block(result, judge_model))
    if args.check:
        if rendered != current:
            print(f"out of date: {args.readme} — run without --check", file=sys.stderr)
            return 1
        print(f"{args.readme} matches {AGREEMENT.relative_to(REPO_ROOT)}")
        return 0
    args.readme.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
