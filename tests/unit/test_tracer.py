"""OTelTracer + NoOpTracer: spans recorded, attributes captured, trace-id bound."""

from __future__ import annotations

import sys
import types

import pytest
import structlog

from verity.observability.tracer import NoOpTracer, OTelTracer, RingBufferSpanExporter
from verity.types import UsageStats, new_id


@pytest.mark.asyncio
async def test_noop_tracer_yields_span_that_swallows_everything() -> None:
    tracer = NoOpTracer()
    trace_id = new_id()
    async with tracer.stage("noop.stage", trace_id) as span:
        span.set_usage(UsageStats(input_tokens=1, output_tokens=1))
        span.set_attribute("k", "v")
        span.record_hits(3, 0.42)


@pytest.mark.asyncio
async def test_otel_tracer_records_span_with_latency_and_attributes() -> None:
    tracer = OTelTracer(service_name="verity-test")
    trace_id = new_id()

    async with tracer.stage("decompose", trace_id) as span:
        span.set_usage(UsageStats(input_tokens=42, output_tokens=7, llm_calls=1, cost_usd=0.0005))
        span.set_attribute("n_sub_questions", 1)

    async with tracer.stage("retrieve.dense", trace_id) as span:
        span.record_hits(8, 0.87)

    spans = tracer.get_finished_spans()
    by_name = {s.name: s for s in spans}
    assert set(by_name) == {"decompose", "retrieve.dense"}

    decompose = by_name["decompose"]
    assert decompose.attributes["verity.trace_id"] == str(trace_id)
    assert decompose.attributes["usage.input_tokens"] == 42
    assert decompose.attributes["usage.cost_usd"] == pytest.approx(0.0005)
    assert decompose.attributes["n_sub_questions"] == 1
    assert decompose.attributes["latency_ms"] >= 0.0

    retrieve = by_name["retrieve.dense"]
    assert retrieve.attributes["retrieval.count"] == 8
    assert retrieve.attributes["retrieval.top_score"] == pytest.approx(0.87)


@pytest.mark.asyncio
async def test_otel_tracer_reset_clears_history() -> None:
    tracer = OTelTracer()
    async with tracer.stage("x", new_id()) as _:
        pass
    assert len(tracer.get_finished_spans()) == 1
    tracer.reset()
    assert tracer.get_finished_spans() == []


@pytest.mark.asyncio
async def test_span_buffer_is_bounded_and_keeps_the_newest() -> None:
    tracer = OTelTracer(max_spans=3)
    for i in range(5):
        async with tracer.stage(f"s{i}", new_id()):
            pass
    assert [s.name for s in tracer.get_finished_spans()] == ["s2", "s3", "s4"]


def test_ring_buffer_rejects_a_non_positive_size() -> None:
    with pytest.raises(ValueError, match="max_spans"):
        RingBufferSpanExporter(0)


def test_configured_endpoint_without_the_otlp_extra_warns_at_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A None entry makes the import raise ImportError, as when the extra is absent.
    monkeypatch.setitem(sys.modules, "opentelemetry.exporter.otlp.proto.http.trace_exporter", None)
    with structlog.testing.capture_logs() as events:
        tracer = OTelTracer(otel_endpoint="http://collector:4318/v1/traces")
    warnings = [e for e in events if e["event"] == "otlp_export_disabled"]
    assert len(warnings) == 1
    assert warnings[0]["log_level"] == "warning"
    assert "verity[otlp]" in warnings[0]["hint"]
    assert tracer.otlp_enabled is False


def test_configured_endpoint_with_the_extra_attaches_a_batched_exporter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

    created: list[str] = []

    class _FakeOTLPSpanExporter(SpanExporter):
        def __init__(self, *, endpoint: str) -> None:
            created.append(endpoint)

        def export(self, spans: object) -> SpanExportResult:  # type: ignore[override]
            return SpanExportResult.SUCCESS

    module = types.ModuleType("trace_exporter")
    module.OTLPSpanExporter = _FakeOTLPSpanExporter  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules, "opentelemetry.exporter.otlp.proto.http.trace_exporter", module
    )
    tracer = OTelTracer(otel_endpoint="http://collector:4318/v1/traces")
    assert tracer.otlp_enabled is True
    assert created == ["http://collector:4318/v1/traces"]
    tracer.shutdown()


def test_no_endpoint_means_no_otlp_exporter() -> None:
    assert OTelTracer().otlp_enabled is False
