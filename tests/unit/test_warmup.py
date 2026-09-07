"""Warmup helper + AgentEvaluator.cold_start_ms reporting."""

from __future__ import annotations

import asyncio
import time

import pytest

from verity.observability.warmup import WARMUP_QUESTION, agent_warmup
from verity.types import Answer, Confidence, UsageStats, new_id


class _SlowFirstCallAgent:
    """Agent that reports a big latency on the first ``answer()`` call, then a small one.

    Simulates the cold-start pattern (model load) that the warmup helper is
    designed to soak up before the timed loop begins.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def answer(self, question: str) -> Answer:
        self.calls.append(question)
        first_call = len(self.calls) == 1
        # Simulated wall-clock work: 30ms on cold-start, 1ms otherwise.
        await asyncio.sleep(0.030 if first_call else 0.001)
        return Answer(
            query=question,
            text="stub",
            confidence=Confidence(score=0.9, refused=False, rationale="stub"),
            usage=UsageStats(latency_ms=30.0 if first_call else 1.0),
            trace_id=new_id(),
        )


@pytest.mark.asyncio
async def test_agent_warmup_returns_measured_cold_start() -> None:
    agent = _SlowFirstCallAgent()
    cold_ms = await agent_warmup(agent)
    assert cold_ms > 20.0, f"expected cold-start ≥ 20ms, got {cold_ms}"
    assert agent.calls == [WARMUP_QUESTION]


@pytest.mark.asyncio
async def test_second_call_after_warmup_is_faster() -> None:
    """The whole point of warmup: subsequent calls run at steady state."""
    agent = _SlowFirstCallAgent()
    await agent_warmup(agent)
    started = time.perf_counter()
    await agent.answer("real question")
    steady_ms = (time.perf_counter() - started) * 1000
    assert steady_ms < 15.0, f"steady-state call should be << cold-start; got {steady_ms}ms"
