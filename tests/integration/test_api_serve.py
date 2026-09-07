"""Integration: full FastAPI app running with a real OTelTracer + a real
in-memory retriever + scripted agent — no pgvector required.

Boots the app via httpx's ASGI transport, issues a couple of real
``agent.answer`` calls to populate the tracer, then asserts the /metrics
endpoints return coherent aggregates. Marked ``integration`` because it
composes the whole stack (ingestion + agent + tracer + FastAPI); the unit
lane exercises each piece in isolation.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

import httpx
import pytest

from verity.agent.agent import create_default_agent
from verity.agent.scripted_stub import build_stub_router
from verity.api import create_app
from verity.ingestion.chunker import StructureAwareChunker
from verity.ingestion.embedder import LocalEmbedder
from verity.ingestion.parsers import create_default_parser
from verity.ingestion.pipeline import IngestionPipeline, discover_sources
from verity.observability.tracer import OTelTracer
from verity.retrieval import CrossEncoderReranker, HybridRetriever, InMemoryVectorStore

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS = REPO_ROOT / "datasets" / "corpus"


class _FakeEmbeddingBackend:
    def __init__(self, dim: int = 32) -> None:
        self._dim = dim

    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> list[list[float]]:
        rows: list[list[float]] = []
        for s in sentences:
            v = [0.0] * self._dim
            for tok in s.lower().split():
                v[hash(tok) % self._dim] += 1.0
            norm = sum(x * x for x in v) ** 0.5 or 1.0
            rows.append([x / norm for x in v])
        return rows

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim


class _FakeRerankerBackend:
    def predict(
        self,
        sentences: Sequence[tuple[str, str]],
        *,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> list[float]:
        return [float(len(set(q.lower().split()) & set(p.lower().split()))) for q, p in sentences]


async def _seed_and_query(tracer: OTelTracer) -> None:
    """One full agent request end-to-end so /metrics has non-zero aggregates."""
    embedder = LocalEmbedder(model="fake", expected_dim=32, backend=_FakeEmbeddingBackend(32))
    store = InMemoryVectorStore()
    pipeline = IngestionPipeline(
        parser=create_default_parser(),
        chunker=StructureAwareChunker(),
        embedder=embedder,
        sink=store,
    )
    result = await pipeline.ingest(discover_sources(str(CORPUS)))
    assert result.failures == []

    retriever = HybridRetriever(
        store=store,
        embedder=embedder,
        reranker=CrossEncoderReranker(model="fake", backend=_FakeRerankerBackend()),
        tracer=tracer,
    )
    agent = create_default_agent(retriever=retriever, client=build_stub_router(), tracer=tracer)
    # Two answerable requests + one out-of-scope so /metrics observes both.
    await agent.answer("What is the notice period for terminating the agreement?")
    await agent.answer("What is the CEO's home address?")


@pytest.mark.asyncio
async def test_metrics_and_dashboard_after_real_requests(tmp_path: Path) -> None:
    tracer = OTelTracer()
    await _seed_and_query(tracer)
    app = create_app(tracer=tracer, runs_dir=tmp_path, dataset="questions")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://verity.test") as client:
        health = await client.get("/health")
        assert health.status_code == 200

        metrics = await client.get("/metrics")
        assert metrics.status_code == 200
        data = metrics.json()
        assert data["n_requests"] == 2
        # One of the two questions is out-of-scope → refused.
        assert data["n_refusals"] >= 1
        stage_names = {s["stage"] for s in data["stages"]}
        # The agent + retriever should have emitted these stages at least.
        for expected in {"agent.decompose", "agent.retrieve", "retrieve.dense", "retrieve.rerank"}:
            assert expected in stage_names, (
                f"expected {expected!r} in stage aggregates, got {stage_names!r}"
            )

        prom = await client.get("/metrics/prom")
        assert prom.status_code == 200
        assert "verity_requests_total 2" in prom.text
        assert "verity_stage_latency_ms" in prom.text

        dashboard = await client.get("/dashboard")
        assert dashboard.status_code == 200
        html = dashboard.text
        assert "Verity dashboard" in html
        assert "retrieve.dense" in html


def test_asyncio_event_loop_bridge_available() -> None:
    # Sanity: pytest-asyncio can wire the test above.
    loop = asyncio.new_event_loop()
    try:
        assert loop.is_running() is False
    finally:
        loop.close()
