"""The figure-rule measurement script reads figures the way its report says."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "measure_figure_rule.py"
RUN_OF_RECORD = Path("datasets/eval/runs/live/eval_d0ca1d5_questions.json")


@pytest.fixture(scope="module")
def rule() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_figure_rule", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_figures_are_numbers_and_dated_months(rule: ModuleType) -> None:
    assert rule.figures("Pay $1,250,000 by March 1, 2024.") == {"1250000", "1", "2024", "march"}
    assert rule.figures("Either party may terminate.") == set()


def test_identifiers_are_not_figures_in_the_exempt_variant(rule: ModuleType) -> None:
    text = "Order Form No. NW-2025-001 for FY2025 lists 250 seats."
    assert rule.figures(text, skip_identifiers=True) == {"250"}


def test_one_arithmetic_step_or_a_sum_of_three_is_derivable(rule: ModuleType) -> None:
    assert rule.derivable(34000, [85000, 40])  # 40% of the fee
    assert rule.derivable(24.6, [71.4, 46.8])
    assert not rule.derivable(155, [75, 40])  # needs 40 twice


def test_the_run_of_record_measures_as_published(rule: ModuleType) -> None:
    """docs/audit/2026-11-03-figure-rule-measurement.md, first row."""
    result = rule.measure(RUN_OF_RECORD)
    assert result["answered"] == 12
    assert result["strict"] == {"q-013": ["00", "02", "06"]}
    assert result["exempt"] == {"q-013": ["00", "02", "06"]}
