"""The naive-baseline comparison applies the pre-registered rule, nothing else."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load() -> ModuleType:
    path = REPO_ROOT / "scripts" / "compare_naive_baseline.py"
    spec = importlib.util.spec_from_file_location("compare_naive_baseline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _m(k: int, n: int, lo: float, hi: float) -> dict[str, object]:
    return {"k": k, "n": n, "value": k / n, "ci95_wilson": [lo, hi]}


def test_the_abstention_patterns_are_the_pre_registered_ones() -> None:
    prereg = (REPO_ROOT / "docs/preregistration/naive-rag-baseline.md").read_text()
    # The block is indented under its list item in the Markdown.
    block = re.search(r"```\n(.*?)\n\s*```", prereg, re.DOTALL)
    assert block is not None
    patterns = tuple(line.strip() for line in block.group(1).splitlines())
    assert patterns == _load().ABSTENTION_PATTERNS


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The documents do not mention a CEO's home address.", True),
        ("The provided evidence doesn't specify the retention period.", True),
        ("There is no information about parking in the documents.", True),
        ("I don't know based on the provided documents.", True),
        ("The notice period is 90 days.", False),
        ("Revenue did not decline; it rose 12 %.", False),
    ],
)
def test_self_abstention(text: str, expected: bool) -> None:
    assert _load().is_self_abstention(text) is expected


def test_overlapping_intervals_read_not_distinguishable() -> None:
    read = _load().read
    assert read(_m(1, 13, 0.01, 0.33), _m(3, 13, 0.08, 0.50)) == "not distinguishable"
    # Touching bounds overlap.
    assert read(_m(1, 13, 0.01, 0.33), _m(9, 13, 0.33, 0.87)) == "not distinguishable"


def test_disjoint_intervals_read_distinguishable_with_verity_direction() -> None:
    read = _load().read
    assert read(_m(1, 13, 0.01, 0.33), _m(13, 13, 0.77, 1.0)) == "distinguishable: Verity lower"
    assert read(_m(13, 13, 0.77, 1.0), _m(1, 13, 0.01, 0.33)) == "distinguishable: Verity higher"


def test_an_empty_denominator_is_not_read() -> None:
    empty = {"k": 0, "n": 0, "value": None, "ci95_wilson": None}
    assert _load().read(empty, _m(1, 13, 0.01, 0.33)) == "not computable (n = 0)"


def test_missing_markers_fail_loudly() -> None:
    with pytest.raises(SystemExit, match="markers"):
        _load().replace_block("no markers\n", "block")
