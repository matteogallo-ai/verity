"""The demo recording's spend cap and its edit (scripts/demo/media.py, record.py)."""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from verity.agent.support import SupportCheck
from verity.config import Settings
from verity.types import RefusalReason

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO = REPO_ROOT / "scripts" / "demo"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("demo_media", DEMO / "media.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["demo_media"] = module  # dataclasses resolve annotations through it
    spec.loader.exec_module(module)
    return module


media = _load()
Wait, Segment = media.Wait, media.Segment


def test_one_answer_admitted_under_the_server_budget_cannot_cross_the_cap() -> None:
    assert media.SERVER_BUDGET_USD + media.PER_ANSWER_BOUND_USD <= media.CAP_USD
    price = Settings().price_for(media.AGENT_MODEL)
    assert (
        media.MAX_INPUT_TOKENS * price.input_usd_per_mtok
        + media.MAX_OUTPUT_TOKENS * price.output_usd_per_mtok
    ) / 1e6 <= media.PER_ANSWER_BOUND_USD
    assert Settings().anthropic_model == media.AGENT_MODEL
    # The bound must dominate the dearest live answer of record.
    costs = [
        row["usage"]["cost_usd"]
        for path in (REPO_ROOT / "datasets/eval/runs/live").glob("eval_*.json")
        for row in json.loads(path.read_text()).get("per_audit", [])
        if isinstance(row.get("usage"), dict)
    ]
    assert costs and max(costs) < media.PER_ANSWER_BOUND_USD


def test_the_compose_overlay_sets_the_server_budget_of_media_py() -> None:
    overlay = yaml.safe_load((DEMO / "compose.demo.yml").read_text())
    env = overlay["services"]["app"]["environment"]
    assert float(env["VERITY_API_BUDGET_USD"]) == media.SERVER_BUDGET_USD
    # Longer than a recording session, so the budget never refills mid-session.
    assert float(env["VERITY_API_BUDGET_WINDOW_S"]) >= 3600
    # A retry may be billed without its usage ever reaching the budget.
    assert env["VERITY_LLM_MAX_RETRIES"] == "0"


def test_check_spend_returns_the_total_under_the_cap() -> None:
    assert media.check_spend([0.02, 0.013, 0.006]) == pytest.approx(0.039)
    assert media.check_spend([]) == 0.0


def test_check_spend_raises_over_the_cap() -> None:
    with pytest.raises(media.SpendCapError, match="exceeds"):
        media.check_spend([0.3, 0.21])


def test_segments_speed_up_only_the_waits() -> None:
    parts = media.segments((0.0, 20.0), [Wait(5.0, 11.0), Wait(14.0, 17.0)], 3.0)
    assert parts == [
        Segment(0.0, 5.0, 1.0),
        Segment(5.0, 11.0, 3.0),
        Segment(11.0, 14.0, 1.0),
        Segment(14.0, 17.0, 3.0),
        Segment(17.0, 20.0, 1.0),
    ]
    assert media.duration(parts) == pytest.approx(5 + 2 + 3 + 1 + 3)


def test_segments_clip_waits_to_the_span_and_drop_those_outside() -> None:
    parts = media.segments((4.0, 10.0), [Wait(1.0, 2.0), Wait(3.0, 6.0), Wait(9.0, 12.0)], 2.0)
    assert parts == [Segment(4.0, 6.0, 2.0), Segment(6.0, 9.0, 1.0), Segment(9.0, 10.0, 2.0)]


@pytest.mark.parametrize(
    ("span", "waits", "speed"),
    [
        ((5.0, 5.0), [], 2.0),
        ((0.0, 10.0), [Wait(1.0, 4.0), Wait(3.0, 5.0)], 2.0),
        ((0.0, 10.0), [], 0.5),
    ],
    ids=["empty-span", "overlapping-waits", "slow-motion"],
)
def test_segments_reject_malformed_input(
    span: tuple[float, float], waits: list[object], speed: float
) -> None:
    with pytest.raises(ValueError):
        media.segments(span, waits, speed)


def test_filter_graph_retimes_each_part_and_concatenates_them() -> None:
    graph = media.filter_graph([Segment(0.0, 1.5, 1.0), Segment(1.5, 4.0, 8.0)], "null[out]")
    assert graph == (
        "[0:v]trim=start=0.000:end=1.500,setpts=(PTS-STARTPTS)/1[v0];"
        "[0:v]trim=start=1.500:end=4.000,setpts=(PTS-STARTPTS)/8[v1];"
        "[v0][v1]concat=n=2:v=1:a=0,null[out]"
    )
    with pytest.raises(ValueError):
        media.filter_graph([], "null[out]")


def _record_constants() -> dict[str, object]:
    """Module-level constants of record.py, read without importing Playwright."""
    tree = ast.parse((DEMO / "record.py").read_text())
    return {
        node.targets[0].id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
    }


def test_the_captions_quote_the_live_settings_and_rationales() -> None:
    constants = _record_constants()
    assert constants["CONFIDENCE_THRESHOLD"] == Settings().confidence_threshold
    injection = SupportCheck(
        coverage=1.0, cited_rerank_score=None, reason=RefusalReason.SUSPECTED_INJECTION
    )
    assert constants["INJECTION_RATIONALE"] == injection.rationale


def test_the_demo_questions_are_labelled_rows_of_the_dataset() -> None:
    constants = _record_constants()
    rows = {
        row["question"]: row
        for row in map(json.loads, (REPO_ROOT / "datasets/eval/questions.jsonl").open())
    }
    assert rows[constants["ANSWERABLE"]]["answerable"] is True
    assert rows[constants["ANSWERABLE_2"]]["answerable"] is True
    assert rows[constants["OUT_OF_CORPUS"]]["answerable"] is False
