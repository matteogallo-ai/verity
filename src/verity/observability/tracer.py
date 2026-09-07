"""Tracer + StageSpan implementations.

Two backends behind the same :class:`Tracer` protocol:

- :class:`NoOpTracer` — the default. Every stage span is a null object that
  swallows ``set_usage`` / ``set_attribute`` / ``record_hits``. Zero overhead,
  zero dependency, and it lets the ~120 existing unit tests keep instantiating
  the agent without an observability stack.

- :class:`OTelTracer` — a real OpenTelemetry SDK tracer. Spans are exported
  through :class:`InMemorySpanExporter` (queryable by ``/metrics`` and the
  dashboard) and, when ``config.otel_endpoint`` is set, additionally forwarded
  to an OTLP collector via a batched OTLP HTTP exporter.

The ``InMemorySpanExporter`` in the OpenTelemetry SDK is exactly what we need:
it accumulates :class:`ReadableSpan`s in memory so the FastAPI ``/metrics``
handler can aggregate latency percentiles + retrieval hit counts + LLM usage
without a collector round-trip. Same data structure feeds the Prometheus text
export and the server-rendered dashboard.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from verity.observability.base import StageSpan
from verity.types import TraceId, UsageStats

if TYPE_CHECKING:  # pragma: no cover — typing only
    from opentelemetry.sdk.trace import ReadableSpan
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


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


class OTelTracer:
    """OpenTelemetry-backed :class:`Tracer`.

    Every process gets a single :class:`InMemorySpanExporter` (the ``exporter``
    attribute) that the FastAPI ``/metrics`` handler reads. When
    ``otel_endpoint`` is set, a second OTLP exporter is added; both share the
    same span pipeline.
    """

    def __init__(
        self,
        *,
        service_name: str = "verity",
        otel_endpoint: str | None = None,
    ) -> None:
        # Local imports keep the observability layer optional at import time —
        # a NoOpTracer user does not need OpenTelemetry to be installed.
        from opentelemetry import trace as otel_trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        self._provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        self.exporter: InMemorySpanExporter = InMemorySpanExporter()
        self._provider.add_span_processor(SimpleSpanProcessor(self.exporter))

        if otel_endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                    OTLPSpanExporter,
                )

                self._provider.add_span_processor(
                    SimpleSpanProcessor(OTLPSpanExporter(endpoint=otel_endpoint))
                )
            except ImportError:  # pragma: no cover — optional extras
                pass

        self._otel_trace = otel_trace
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
        return list(self.exporter.get_finished_spans())

    def reset(self) -> None:
        """Drop accumulated spans — useful in tests and between demo sessions."""
        self.exporter.clear()


__all__ = ["NoOpTracer", "OTelStageSpan", "OTelTracer"]
