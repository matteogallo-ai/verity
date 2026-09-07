"""Observability layer: tracer, structured-logging plumbing, warmup helper."""

from __future__ import annotations

from verity.observability.base import StageSpan, Tracer
from verity.observability.logging import (
    bind_trace,
    configure_logging,
    current_trace_id,
    request_trace,
    unbind_trace,
)
from verity.observability.tracer import NoOpTracer, OTelStageSpan, OTelTracer
from verity.observability.warmup import WARMUP_QUESTION, agent_warmup

__all__ = [
    "WARMUP_QUESTION",
    "NoOpTracer",
    "OTelStageSpan",
    "OTelTracer",
    "StageSpan",
    "Tracer",
    "agent_warmup",
    "bind_trace",
    "configure_logging",
    "current_trace_id",
    "request_trace",
    "unbind_trace",
]
