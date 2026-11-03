"""The blind judge validation is published once and never edited.

``datasets/eval/judge_validation/SHA256SUMS`` pins the blind human labels, the answer
key, the judge's verdicts on perturbed claims and the agreement computed from them.
An adjudication is published in its own files; it never rewrites these.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

VALIDATION = Path(__file__).resolve().parents[2] / "datasets" / "eval" / "judge_validation"
BLIND = {
    "annotate_morning_1.csv",
    "annotate_morning_2.csv",
    "judge_key.json",
    "perturbed_verdicts.json",
    "claims_annotated.csv",
    "agreement.json",
}


def _pinned() -> dict[str, str]:
    lines = (VALIDATION / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    return {name: digest for digest, name in (line.split() for line in lines if line.strip())}


def test_every_blind_result_is_pinned() -> None:
    assert BLIND <= set(_pinned())


@pytest.mark.parametrize("name", sorted(_pinned()))
def test_a_pinned_validation_file_never_changes(name: str) -> None:
    digest = hashlib.sha256((VALIDATION / name).read_bytes()).hexdigest()
    assert digest == _pinned()[name], f"{name} was modified; the blind validation is immutable"
