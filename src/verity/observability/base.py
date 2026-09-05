"""Observability contracts: trace every query end-to-end, per stage.

Each pipeline stage (parse, embed, dense search, sparse search, fusion, rerank,
decompose, synthesize, judge) opens a span recording latency, token/cost usage where
relevant, and the hit scores it produced. Spans are exported via OpenTelemetry and the
same data backs the ``/metrics`` endpoint and the dashboard (cost, latency, quality
over time). Logging is structured (structlog) and carries the ``trace_id`` on every line.

Implementations land in S5.
"""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from typing import Protocol, runtime_checkable

from verity.types import TraceId, UsageStats


@runtime_checkable
class StageSpan(Protocol):
    """A single stage within a query trace."""

    def set_usage(self, usage: UsageStats) -> None: ...

    def set_attribute(self, key: str, value: str | int | float | bool) -> None: ...

    def record_hits(self, count: int, top_score: float | None) -> None: ...


@runtime_checkable
class Tracer(Protocol):
    """Opens stage spans bound to a query's ``trace_id``."""

    def stage(self, name: str, trace_id: TraceId) -> AbstractAsyncContextManager[StageSpan]: ...
