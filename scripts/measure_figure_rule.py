"""Measure a figure rule for the support check on existing run artefacts, without
activating it and without any model call.

The rule under measurement: an answered draft is refused when its answer contains a
figure (a number, or a month next to a day number) that none of its cited quotes
contains. Two variants are measured:

- ``strict``: every figure of the answer must be in a quote;
- ``exempt``: a figure is also accepted when it is in the question, when it is part
  of an identifier ("FY2025", "No. 1", "NW-2025-001"), or when one arithmetic step
  (+, -, x, /, a percentage) or the sum of three numbers from the quotes and the
  question gives it.

For each artefact, the answered (non-refused) examples of ``per_audit`` are checked
and the ones each variant would refuse are listed with the figures it would miss.
Whether a refused answer was correct is read by a person against the reference
answer; the script does not judge.

Usage: ``python scripts/measure_figure_rule.py ARTEFACT.json [...]``.
"""

from __future__ import annotations

import itertools
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
# A number not glued to letters or hyphens, so "FY2025" and "NW-2025-001" are names.
_STANDALONE_RE = re.compile(r"(?<![A-Za-z0-9\-])\d+(?:[.,]\d+)*(?![A-Za-z\-]|\d)")
_THOUSANDS_RE = re.compile(r",(?=\d{3}(?:\D|$))")
_MONTH = (
    r"(january|february|march|april|may|june|july|august|september|october|november"
    r"|december)"
)
# A month counts only next to a day number, so the verb "may" is not a date.
_MONTH_RE = re.compile(rf"\b{_MONTH}\s+\d{{1,2}}\b|\b\d{{1,2}}\s+{_MONTH}\b", re.IGNORECASE)
_LABEL_BEFORE_RE = re.compile(
    r"\b(?:no|section|schedule|clause|article|amendment)\.?\s+$", re.IGNORECASE
)


def _normal(number: str) -> str:
    return _THOUSANDS_RE.sub("", number).rstrip(".,")


def _months(text: str) -> set[str]:
    return {(before or after).lower() for before, after in _MONTH_RE.findall(text)}


def figures(text: str, *, skip_identifiers: bool = False) -> set[str]:
    """Numbers (thousands separators dropped) and dated month names in ``text``."""
    if not skip_identifiers:
        return {_normal(n) for n in _NUMBER_RE.findall(text)} | _months(text)
    found = set()
    for match in _STANDALONE_RE.finditer(text):
        if _LABEL_BEFORE_RE.search(text[max(0, match.start() - 15) : match.start()]):
            continue
        found.add(_normal(match.group()))
    return found | _months(text)


def _as_number(figure: str) -> float | None:
    try:
        return float(figure)
    except ValueError:
        return None


def derivable(target: float, pool: Iterable[float]) -> bool:
    """Whether one arithmetic step on two numbers of ``pool``, or the sum of three,
    gives ``target`` (to rounding)."""
    values = list(pool)
    tolerance = max(0.051, abs(target) * 0.005)
    for a, b in itertools.permutations(values, 2):
        candidates = [a + b, a - b, a * b, a * b / 100]
        if b:
            candidates += [a / b, a / b * 100]
        if any(abs(v - target) <= tolerance for v in candidates):
            return True
    return any(
        abs(a + b + c - target) <= tolerance for a, b, c in itertools.combinations(values, 3)
    )


def missing_figures(answer: str, quotes: list[str], question: str, *, variant: str) -> set[str]:
    quoted_text = " ".join(quotes)
    if variant == "strict":
        return figures(answer) - figures(quoted_text)
    quoted = figures(quoted_text, skip_identifiers=True)
    asked = figures(question, skip_identifiers=True)
    missing = figures(answer, skip_identifiers=True) - quoted - asked
    pool = [n for n in (_as_number(f) for f in quoted | asked) if n is not None]
    return {f for f in missing if (n := _as_number(f)) is None or not derivable(n, pool)}


def measure(artefact: Path) -> dict[str, object]:
    audits = json.loads(artefact.read_text(encoding="utf-8"))["per_audit"]
    answered = [a for a in audits if not a["refused"]]
    out: dict[str, object] = {"artefact": artefact.as_posix(), "answered": len(answered)}
    for variant in ("strict", "exempt"):
        refused = {}
        for a in answered:
            quotes = [c["quote"] for c in a["citations"]]
            missing = missing_figures(a["answer_text"], quotes, a["question"], variant=variant)
            if missing:
                refused[a["example_id"]] = sorted(missing)
        out[variant] = refused
    return out


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    for path in argv:
        print(json.dumps(measure(Path(path)), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
