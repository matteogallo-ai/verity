"""Tracer + StageSpan implementations.

Two backends behind the same :class:`Tracer` protocol:

- :class:`NoOpTracer` — the default. Every stage span is a null object that
  swallows ``set_usage`` / ``set_attribute`` / ``record_hits``. Zero overhead,
  zero dependency, and it lets the ~120 existing unit tests keep instantiating
  the agent without an observability stack.

- :class:`OTelTracer` — a real OpenTelemetry SDK tracer. Spans are kept in a
  :class:`RingBufferSpanExporter` (queryable by ``/metrics`` and the dashboard)
  and, when ``config.otel_endpoint`` is set and the ``otlp`` extra is installed,
  also sent to an OTLP collector through a batching processor.

The ring buffer holds the newest ``max_spans`` spans: ``/metrics`` aggregates over
that window, and a long-running server's memory stays bounded. The SDK's
``InMemorySpanExporter`` keeps every span forever.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

import structlog
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from verity.observability.base import StageSpan
from verity.types import TraceId, UsageStats

log = structlog.get_logger(__name__)

DEFAULT_MAX_SPANS = 10_000


# --------------------------------------------------------------------------------------
# NoOp
# --------------------------------------------------------------------------------------


class _NoOpSpan:
    """Structural :class:`StageSpan` that does nothing. Zero overhead."""

    def set_usage(self, usage: UsageStats) -> None:
        return

    def set_attribute(self, key: str, value: str | int | float | bool) -> None:
        return

    def record_hits(self, count: int, top_score: float | None) -> None:
        return


class NoOpTracer:
    """Default Tracer: no spans recorded. Used everywhere unless the caller
    (CLI/API) provides a real :class:`OTelTracer`."""

    @asynccontextmanager
    async def stage(self, name: str, trace_id: TraceId) -> AsyncIterator[StageSpan]:
        yield _NoOpSpan()


# --------------------------------------------------------------------------------------
# OTel
# --------------------------------------------------------------------------------------


class OTelStageSpan:
    """Wraps an OTel SDK Span behind the :class:`StageSpan` protocol.

    ``set_usage`` flattens :class:`UsageStats` into structured span attributes so
    the InMemory exporter can aggregate them without knowing about the type.
    """

    def __init__(self, span: Any) -> None:
        self._span = span

    def set_usage(self, usage: UsageStats) -> None:
        self._span.set_attribute("usage.input_tokens", usage.input_tokens)
        self._span.set_attribute("usage.output_tokens", usage.output_tokens)
        self._span.set_attribute("usage.llm_calls", usage.llm_calls)
        self._span.set_attribute("usage.cost_usd", usage.cost_usd)
        self._span.set_attribute("usage.latency_ms", usage.latency_ms)

    def set_attribute(self, key: str, value: str | int | float | bool) -> None:
        self._span.set_attribute(key, value)

    def record_hits(self, count: int, top_score: float | None) -> None:
        self._span.set_attribute("retrieval.count", count)
        if top_score is not None:
            self._span.set_attribute("retrieval.top_score", top_score)


class RingBufferSpanExporter(SpanExporter):
    """Keeps the newest ``max_spans`` finished spans in memory, dropping the oldest."""

    def __init__(self, max_spans: int = DEFAULT_MAX_SPANS) -> None:
        if max_spans < 1:
            raise ValueError(f"max_spans must be >= 1, got {max_spans}")
        self._spans: deque[ReadableSpan] = deque(maxlen=max_spans)
        # The OTel span processor may export from another thread than the reader.
        self._lock = threading.Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        with self._lock:
            self._spans.extend(spans)
        return SpanExportResult.SUCCESS

    def get_finished_spans(self) -> list[ReadableSpan]:
        with self._lock:
            return list(self._spans)

    def clear(self) -> None:
        with self._lock:
            self._spans.clear()


class OTelTracer:
    """OpenTelemetry-backed :class:`Tracer`.

    Every process gets one :class:`RingBufferSpanExporter` (the ``exporter``
    attribute) that the FastAPI ``/metrics`` handler reads. With ``otel_endpoint``
    set, an OTLP exporter is added behind a batching processor so a slow collector
    never sits on the request path; ``otlp_enabled`` says whether it is active.
    """

    def __init__(
        self,
        *,
        service_name: str = "verity",
        otel_endpoint: str | None = None,
        max_spans: int = DEFAULT_MAX_SPANS,
    ) -> None:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

        self._provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        self.exporter = RingBufferSpanExporter(max_spans)
        self._provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        self.otlp_enabled = False

        if otel_endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                    OTLPSpanExporter,
                )
            except ImportError:
                log.warning(
                    "otlp_export_disabled",
                    endpoint=otel_endpoint,
                    hint="VERITY_OTEL_ENDPOINT is set but the OTLP exporter is not "
                    "installed: install verity[otlp] (uv sync --extra otlp).",
                )
            else:
                self._provider.add_span_processor(
                    BatchSpanProcessor(OTLPSpanExporter(endpoint=otel_endpoint))
                )
                self.otlp_enabled = True

        self._tracer = self._provider.get_tracer("verity.observability")

    @asynccontextmanager
    async def stage(self, name: str, trace_id: TraceId) -> AsyncIterator[StageSpan]:
        started = time.perf_counter()
        with self._tracer.start_as_current_span(name) as otel_span:
            # Bind the query's trace id as an attribute so exported spans can
            # be grouped by request even though OTel manages its own span-id.
            otel_span.set_attribute("verity.trace_id", str(trace_id))
            wrapped = OTelStageSpan(otel_span)
            try:
                yield wrapped
            finally:
                elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
                otel_span.set_attribute("latency_ms", elapsed_ms)

    def get_finished_spans(self) -> list[ReadableSpan]:
        """Convenience: raw spans the API layer can aggregate. Never blocks."""
        return self.exporter.get_finished_spans()

    def reset(self) -> None:
        """Drop accumulated spans — useful in tests and between demo sessions."""
        self.exporter.clear()

    def shutdown(self) -> None:
        """Flush and stop every span processor (the OTLP batch included)."""
        self._provider.shutdown()


__all__ = ["NoOpTracer", "OTelStageSpan", "OTelTracer", "RingBufferSpanExporter"]
