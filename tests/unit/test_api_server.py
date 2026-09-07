"""FastAPI /metrics + /metrics/prom + /dashboard — hermetic (tmp_path runs dir).

Populates the shared OTelTracer with a couple of spans before hitting /metrics
so the JSON aggregates are non-trivial. Uses FastAPI's TestClient so no port is
actually bound.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from verity.api import create_app
from verity.observability.tracer import OTelTracer
from verity.types import UsageStats, new_id


def _seed_tracer_with_a_query(tracer: OTelTracer) -> None:
    """Emit spans that mimic one full agent request (retrieve + confidence)."""
    trace_id = new_id()

    async def _fake_request() -> None:
        async with tracer.stage("retrieve.dense", trace_id) as span:
            span.record_hits(8, 0.87)
        async with tracer.stage("agent.confidence", trace_id) as span:
            span.set_attribute("refused", False)
            span.set_usage(UsageStats(cost_usd=0.0003, input_tokens=100, output_tokens=20))

    asyncio.run(_fake_request())


def test_health_endpoint_returns_ok(tmp_path: Path) -> None:
    app = create_app(tracer=None, runs_dir=tmp_path, dataset="questions")
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_metrics_endpoint_empty_when_no_spans(tmp_path: Path) -> None:
    app = create_app(tracer=None, runs_dir=tmp_path, dataset="questions")
    client = TestClient(app)
    r = client.get("/metrics")
    assert r.status_code == 200
    data = r.json()
    assert data["n_requests"] == 0
    assert data["refusal_rate"] == 0.0
    assert data["stages"] == []


def test_metrics_json_aggregates_are_populated(tmp_path: Path) -> None:
    tracer = OTelTracer()
    _seed_tracer_with_a_query(tracer)
    app = create_app(tracer=tracer, runs_dir=tmp_path, dataset="questions")
    client = TestClient(app)
    r = client.get("/metrics")
    assert r.status_code == 200
    data = r.json()
    assert data["n_requests"] == 1
    assert data["n_refusals"] == 0
    # Two stages seeded.
    stages = {s["stage"]: s for s in data["stages"]}
    assert "retrieve.dense" in stages
    assert "agent.confidence" in stages
    assert stages["retrieve.dense"]["p50_ms"] >= 0.0
    assert data["total_cost_usd"] > 0.0


def test_metrics_prom_returns_prometheus_text(tmp_path: Path) -> None:
    tracer = OTelTracer()
    _seed_tracer_with_a_query(tracer)
    app = create_app(tracer=tracer, runs_dir=tmp_path, dataset="questions")
    client = TestClient(app)
    r = client.get("/metrics/prom")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    body = r.text
    assert "verity_requests_total 1" in body
    assert "verity_stage_latency_ms" in body


def test_dashboard_renders_html(tmp_path: Path) -> None:
    tracer = OTelTracer()
    _seed_tracer_with_a_query(tracer)
    app = create_app(tracer=tracer, runs_dir=tmp_path, dataset="questions")
    client = TestClient(app)
    r = client.get("/dashboard")
    assert r.status_code == 200
    assert "Verity dashboard" in r.text
    # Should mention the stage names we seeded.
    assert "retrieve.dense" in r.text
    assert "agent.confidence" in r.text
    # And gracefully handle the empty-run-store case.
    assert "No persisted runs found" in r.text
