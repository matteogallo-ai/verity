"""LLMConfidenceScorer: threshold decision + malformed-output fail-open (refuse)."""

from __future__ import annotations

import json

import pytest

from verity.agent.scorer import LLMConfidenceScorer
from verity.llm.clients import StubLLMClient
from verity.llm.routing import MultiProviderRoutingClient


def _router(text: str) -> MultiProviderRoutingClient:
    return MultiProviderRoutingClient([StubLLMClient(lambda _: text)])


@pytest.mark.asyncio
async def test_above_threshold_not_refused() -> None:
    scorer = LLMConfidenceScorer(
        _router(json.dumps({"score": 0.9, "refused": False, "rationale": "strong"})),
        threshold=0.55,
    )
    c = await scorer.score("q?", "answer", hits=[])
    assert not c.refused
    assert c.score == 0.9
    assert c.rationale == "strong"


@pytest.mark.asyncio
async def test_below_threshold_refused_regardless_of_model_flag() -> None:
    scorer = LLMConfidenceScorer(
        _router(json.dumps({"score": 0.2, "refused": False, "rationale": "weak"})),
        threshold=0.55,
    )
    c = await scorer.score("q?", "answer", hits=[])
    assert c.refused
    assert c.rationale == "weak"


@pytest.mark.asyncio
async def test_model_refusal_honoured_above_threshold() -> None:
    scorer = LLMConfidenceScorer(
        _router(json.dumps({"score": 0.8, "refused": True, "rationale": "conflict"})),
        threshold=0.5,
    )
    c = await scorer.score("q?", "answer", hits=[])
    assert c.refused
    assert c.rationale == "conflict"


@pytest.mark.asyncio
async def test_malformed_output_forces_refusal() -> None:
    scorer = LLMConfidenceScorer(_router("definitely not JSON"), threshold=0.5)
    c = await scorer.score("q?", "answer", hits=[])
    assert c.refused
    assert c.score == 0.0
    assert "malformed" in c.rationale.lower()


@pytest.mark.asyncio
async def test_score_clamped_to_unit_interval() -> None:
    scorer = LLMConfidenceScorer(
        _router(json.dumps({"score": 5.7, "refused": False, "rationale": "ok"})),
        threshold=0.5,
    )
    c = await scorer.score("q?", "answer", hits=[])
    assert c.score == 1.0
    assert not c.refused
