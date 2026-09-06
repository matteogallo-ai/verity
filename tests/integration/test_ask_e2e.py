"""Integration: verity ask end-to-end on pgvector with a stubbed LLM.

Proves the feature phare on the real corpus:
- q-001 → an answer citing the §1.2 termination clause with confidence above threshold.
- q-004 / q-006 → refusal, no fabricated text.

The LLM is the same scripted responder the CLI ``--stub`` flag uses — no keys, no
network — so CI is free and reproducible.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from verity.agent.agent import create_default_agent
from verity.agent.scripted_stub import build_stub_router
from verity.config import get_settings
from verity.db import apply_pending
from verity.ingestion import (
    IngestionPipeline,
    LocalEmbedder,
    StructureAwareChunker,
    create_default_parser,
)
from verity.ingestion.pipeline import discover_sources
from verity.retrieval import CrossEncoderReranker, HybridRetriever, PgVectorStore

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = REPO_ROOT / "datasets" / "corpus"


def _database_url() -> str:
    return os.environ.get("VERITY_DATABASE_URL", get_settings().database_url)


@pytest.fixture(scope="module")
def seeded_retriever() -> HybridRetriever:
    url = _database_url()
    apply_pending(url)
    store = PgVectorStore(database_url=url)

    import asyncio

    async def seed() -> None:
        embedder = LocalEmbedder()
        pipeline = IngestionPipeline(
            parser=create_default_parser(),
            chunker=StructureAwareChunker(),
            embedder=embedder,
            sink=store,
        )
        result = await pipeline.ingest(discover_sources(str(CORPUS_DIR)))
        assert result.failures == []

    asyncio.run(seed())
    return HybridRetriever(
        store=store,
        embedder=LocalEmbedder(),
        reranker=CrossEncoderReranker(),
    )


@pytest.mark.asyncio
async def test_answerable_question_cites_termination_clause(
    seeded_retriever: HybridRetriever,
) -> None:
    agent = create_default_agent(retriever=seeded_retriever, client=build_stub_router())
    answer = await agent.answer("What is the notice period for terminating the agreement?")
    assert not answer.confidence.refused, answer.confidence.rationale
    assert answer.citations, "expected ≥1 citation"
    assert any("thirty" in c.quote.lower() for c in answer.citations)
    for c in answer.citations:
        chunk = next((h.chunk for h in answer.hits_used if h.chunk.id == c.chunk_id), None)
        assert chunk is not None
        assert c.quote in chunk.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "question",
    ["What is the CEO's home address?", "What is the company's policy on Martian mining rights?"],
)
async def test_out_of_scope_questions_refuse(
    seeded_retriever: HybridRetriever, question: str
) -> None:
    agent = create_default_agent(retriever=seeded_retriever, client=build_stub_router())
    answer = await agent.answer(question)
    assert answer.confidence.refused, f"expected refusal on {question!r}"
    assert answer.citations == ()
    assert "don't know" in answer.text.lower()
