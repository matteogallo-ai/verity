"""Warmup helper — factor out cold-start latency from the reported percentiles.

The first ``verity ask`` or ``verity eval run`` invocation pays for
sentence-transformers + cross-encoder model loads (multi-second overhead visible
as p95/p99 outliers in the S4 scorecard). Those numbers are real, but they
describe *first-boot latency*, not steady-state serving.

:func:`agent_warmup` runs one full ``agent.answer("warmup ...")`` on a
throwaway question so every lazy component is warm before the timed loop begins.
The wall-clock of that single call is the ``cold_start_ms`` — reported in the
Scorecard as a distinct field so honesty is preserved (nothing is hidden, it is
just placed where it belongs).
"""

from __future__ import annotations

import time

from verity.agent.base import Agent

WARMUP_QUESTION = "warmup: any short factual question suffices"


async def agent_warmup(agent: Agent) -> float:
    """Trigger all lazy model loads by running one agent request. Return the
    measured wall-clock cold-start latency in milliseconds."""
    started = time.perf_counter()
    await agent.answer(WARMUP_QUESTION)
    return round((time.perf_counter() - started) * 1000, 3)


__all__ = ["WARMUP_QUESTION", "agent_warmup"]
