"""Write the README's judge-validation section from the committed agreement artefact.

    uv run python scripts/render_judge_validation.py          # rewrite the block
    uv run python scripts/render_judge_validation.py --check  # CI: README == artefact

The block sits between ``<!-- JUDGE:START -->`` and ``<!-- JUDGE:END -->`` and is
rendered from ``datasets/eval/judge_validation/agreement.json``: the blind human
labels are the reference, and nothing in the block is typed by hand. A second block,
between ``<!-- JUDGE-ADJUDICATED:START -->`` and ``<!-- JUDGE-ADJUDICATED:END -->``,
sets the adjudicated figures of ``agreement_adjudicated.json`` beside the blind ones.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity.ingestion.ids import document_id_for_uri

REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATION = REPO_ROOT / "datasets" / "eval" / "judge_validation"
AGREEMENT = VALIDATION / "agreement.json"
ADJUDICATED = VALIDATION / "agreement_adjudicated.json"
REJUDGED = VALIDATION / "perturbed_verdicts.json"
KEY = VALIDATION / "judge_key.json"
JUDGE_PROMPT = REPO_ROOT / "prompts" / "judge_faithfulness.prompt"
# The corpus note written to carry an injection (adversarial eval): a claim whose answer
# retrieved it put that text, undelimited, in front of the v0.1.0 judge.
INJECTED_DOCUMENT = "datasets/corpus/vendor_onboarding_note.md"
README = REPO_ROOT / "README.md"
START, END = "<!-- JUDGE:START -->", "<!-- JUDGE:END -->"
ADJ_START, ADJ_END = "<!-- JUDGE-ADJUDICATED:START -->", "<!-- JUDGE-ADJUDICATED:END -->"
# Below this many items in the smaller human class, one or two disagreements move κ
# far enough that its value says little: the figure is published as fragile.
FRAGILE_BELOW = 5


@dataclass(frozen=True)
class Provenance:
    """Where the validated claims came from and which judge prompt judged them."""

    real_by_run: dict[str, int]
    perturbed: int
    prompt_version: str
    prompt_sha256: str
    exposed_to_injection: int


def provenance(key: Path = KEY, prompt: Path = JUDGE_PROMPT) -> Provenance:
    """Read from the key and the runs it points at. Refuses when the judge prompt on disk
    is not the one the runs recorded: κ belongs to the prompt that was measured."""
    items = list(json.loads(key.read_text(encoding="utf-8"))["items"].values())
    runs = {
        path: json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))
        for path in sorted({item["run"] for item in items})
    }
    measured = {run["prompt_sha256"]["judge_faithfulness"] for run in runs.values()}
    if len(measured) != 1:
        raise SystemExit(f"the validated runs used different judge prompts: {sorted(measured)}")
    sha = measured.pop()
    text = prompt.read_text(encoding="utf-8")
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != sha:
        raise SystemExit(
            f"{prompt.name} changed since κ was measured (runs recorded {sha[:12]}); "
            "a new judge prompt needs its own rejudge and κ"
        )
    version = re.search(r"^@version (\S+)$", text, re.MULTILINE)
    if version is None:
        raise SystemExit(f"{prompt.name} has no @version line")
    # identity_for resolves a relative path against the working directory, so the
    # path is anchored at the repo root: ingestion ids corpus files by absolute path.
    injected = str(document_id_for_uri(str(REPO_ROOT / INJECTED_DOCUMENT)))
    audits = {path: {a["example_id"]: a for a in run["per_audit"]} for path, run in runs.items()}
    exposed = sum(
        any(
            hit["chunk"]["document_id"] == injected
            for hit in audits[item["run"]][item["source_claim"].split("/")[1]]["hits_used"]
        )
        for item in items
    )
    real = Counter(Path(item["run"]).name for item in items if item["origin"] == "real")
    return Provenance(
        real_by_run=dict(sorted(real.items())),
        perturbed=sum(item["origin"] == "perturbed" for item in items),
        prompt_version=version.group(1),
        prompt_sha256=sha,
        exposed_to_injection=exposed,
    )


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


def _real_origin(prov: Provenance) -> str:
    split = {"6ec544d_dev": "dev", "969e1d5_test": "test"}
    parts = []
    for run, n in prov.real_by_run.items():
        stem = run.removeprefix("eval_").removesuffix(".json")
        parts.append(f"{n} from the {split.get(stem, stem)} run")
    return f"{sum(prov.real_by_run.values())} real claims ({', '.join(parts)})"


def render_block(result: dict[str, Any], judge_model: str, prov: Provenance) -> str:
    s = result["sensitivity"]
    by_kind = " · ".join(f"{k} {v['k']}/{v['n']}" for k, v in sorted(s["by_kind"].items()))
    mixed, real = result["agreement_mixed"], result["agreement_real_only"]
    c = mixed["confusion"]
    lines = [
        f"Judge `{judge_model}` with judge prompt v{prov.prompt_version} "
        f"(`prompts/judge_faithfulness.prompt`, sha256 `{prov.prompt_sha256[:12]}`) against "
        f"**blind human labels**, the reference: {mixed['n_claims']} claims labelled without "
        f"origin or judge verdict, {_real_origin(prov)} and "
        f"{s['judge_detected']['n']} claims perturbed to be false "
        "(one number, date, negation or entity changed). The κ below belongs to this "
        "prompt version: a changed judge prompt is a new judge and needs its own κ.",
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
        f"- robustness to injected content is not measured: judge prompt "
        f"v{prov.prompt_version} puts the retrieved text in front of the judge without "
        "delimiters or an untrusted-content instruction "
        "([THREAT_MODEL](docs/THREAT_MODEL.md#llm01-prompt-injection)), and only "
        f"{prov.exposed_to_injection} of the {mixed['n_claims']} claims had the corpus note "
        "written to carry an injection in the judge's context.",
        "",
        "Generated from "
        "[`datasets/eval/judge_validation/agreement.json`](datasets/eval/judge_validation/agreement.json) "
        "(`scripts/render_judge_validation.py --check`); the labels and the key are pinned in "
        "`datasets/eval/judge_validation/SHA256SUMS`.",
    ]
    return "\n".join(lines)


def render_adjudicated_block(result: dict[str, Any], blind: dict[str, Any]) -> str:
    """The adjudicated figures, next to the blind ones they never replace."""
    adj = result["adjudication"]
    cats = adj["perturbed_judged_true_blind"]
    mixed, real = result["agreement_mixed"], result["agreement_real_only"]

    def kappa(a: dict[str, Any]) -> str:
        return "undefined" if a["cohen_kappa"] is None else f"{a['cohen_kappa']:.2f}"

    s, sb = result["sensitivity"], blind["sensitivity"]
    lines = [
        f"{adj['n_adjudicated']} claims re-read with both verdicts visible, by the annotator "
        f"of the blind labels: every disagreement, and every perturbed claim judged true "
        f"blind. {adj['n_changed']} labels changed, **{adj['n_changed_to_judge']} of them to "
        "the judge's verdict**. The judge's verdict was in view, so a change towards it may "
        "be persuasion as much as correction: these figures bound agreement from above and "
        "validate nothing. Of the "
        f"{cats['n']} perturbed claims judged true blind, {cats['perturbation stays true']} "
        f"stay true after the change and {cats['annotator error']} were annotator misses.",
        "",
        "| Measure | Blind (of record) | Adjudicated |",
        "|---|---|---|",
        f"| Cohen's κ, all claims | {kappa(blind['agreement_mixed'])} | {kappa(mixed)} |",
        f"| Cohen's κ, real claims only (fragile: {_human_counts(real)[1]} unsupported) | "
        f"{kappa(blind['agreement_real_only'])} | {kappa(real)} |",
        f"| Observed agreement, all claims | {_ci(blind['agreement_mixed']['observed'])} | "
        f"{_ci(mixed['observed'])} |",
        f"| Perturbed claims the human finds unsupported | {_ci(sb['human_unsupported'])} | "
        f"{_ci(s['human_unsupported'])} |",
        f"| … of which the judge finds unsupported | "
        f"{_ci(sb['judge_detected_among_human_unsupported'])} | "
        f"{_ci(s['judge_detected_among_human_unsupported'])} |",
        "",
        "Generated from "
        "[`datasets/eval/judge_validation/agreement_adjudicated.json`]"
        "(datasets/eval/judge_validation/agreement_adjudicated.json); each decision and its "
        "reason is in `adjudication.csv`.",
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
    adjudicated = json.loads(ADJUDICATED.read_text(encoding="utf-8"))
    current = args.readme.read_text(encoding="utf-8")
    rendered = replace_block(current, render_block(result, judge_model, provenance()))
    rendered = replace_block(
        rendered, render_adjudicated_block(adjudicated, result), ADJ_START, ADJ_END
    )
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
