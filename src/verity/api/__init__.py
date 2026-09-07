"""HTTP API — /ingest + /ask + /status + /metrics + /dashboard via FastAPI."""

from __future__ import annotations

from verity.api.aggregator import SessionMetrics, aggregate_spans, to_prometheus
from verity.api.models import (
    AnswerView,
    AskResponse,
    CitationView,
    ConfidenceView,
    CorpusSummary,
    IngestResponse,
    Provenance,
    RetrievedChunkView,
    StatusResponse,
    UsageStatsView,
    answer_to_view,
)
from verity.api.router import router
from verity.api.server import create_app
from verity.api.state import AppRuntime, IngestedSource, default_agent_mode

__all__ = [
    "AnswerView",
    "AppRuntime",
    "AskResponse",
    "CitationView",
    "ConfidenceView",
    "CorpusSummary",
    "IngestResponse",
    "IngestedSource",
    "Provenance",
    "RetrievedChunkView",
    "SessionMetrics",
    "StatusResponse",
    "UsageStatsView",
    "aggregate_spans",
    "answer_to_view",
    "create_app",
    "default_agent_mode",
    "router",
    "to_prometheus",
]
