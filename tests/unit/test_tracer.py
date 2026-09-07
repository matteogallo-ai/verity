"""OTelTracer + NoOpTracer: spans recorded, attributes captured, trace-id bound."""

from __future__ import annotations

import pytest

from verity.observability.tracer import NoOpTracer, OTelTracer
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
