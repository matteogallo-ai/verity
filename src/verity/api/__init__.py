"""HTTP API — /metrics + /dashboard exposed via FastAPI."""

from __future__ import annotations

from verity.api.aggregator import SessionMetrics, aggregate_spans, to_prometheus
from verity.api.server import create_app

__all__ = ["SessionMetrics", "aggregate_spans", "create_app", "to_prometheus"]
