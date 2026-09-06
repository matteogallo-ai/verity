"""Integration: HybridRetriever against a live pgvector database.

Ingests the shipped corpus with the real LocalEmbedder + real ingestion pipeline,
then asserts that each answerable question's gold chunk lands in the top-rerank_k of
the retriever. This is the pipeline-level sanity that CI exercises on every push.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from verity.config import get_settings
from verity.db import apply_pending
from verity.ingestion import (
    IngestionPipeline,
    LocalEmbedder,
    StructureAwareChunker,
    create_default_parser,
)
from verity.ingestion.pipeline import discover_sources
from verity.retrieval import (
    CrossEncoderReranker,
    HybridRetriever,
    PgVectorStore,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = REPO_ROOT / "datasets" / "corpus"
DATASET_PATH = REPO_ROOT / "datasets" / "eval" / "questions.jsonl"


def _database_url() -> str:
    return os.environ.get("VERITY_DATABASE_URL", get_settings().database_url)


@pytest.fixture(scope="module")
def seeded_store() -> PgVectorStore:
    url = _database_url()
    apply_pending(url)
    store = PgVectorStore(database_url=url)

    import asyncio

    async def seed() -> None:
        embedder = LocalEmbedder()
        parser = create_default_parser()
        chunker = StructureAwareChunker()
        pipeline = IngestionPipeline(parser=parser, chunker=chunker, embedder=embedder, sink=store)
        result = await pipeline.ingest(discover_sources(str(CORPUS_DIR)))
        assert result.failures == []

    asyncio.run(seed())
    return store


def _answerable_questions() -> list[dict]:  # type: ignore[type-arg]
    with DATASET_PATH.open("r", encoding="utf-8") as fh:
        return [
            json.loads(line)
            for line in fh
            if line.strip()
            and json.loads(line).get("answerable")
            and json.loads(line).get("relevant_chunk_ids")
        ]


@pytest.mark.asyncio
async def test_gold_chunk_in_top_rerank_k(seeded_store: PgVectorStore) -> None:
    retriever = HybridRetriever(
        store=seeded_store,
        embedder=LocalEmbedder(),
        reranker=CrossEncoderReranker(),
    )
    settings = get_settings()
    for record in _answerable_questions():
        hits = await retriever.retrieve(record["question"], k=settings.rerank_k)
        got_ids = {str(h.chunk.id) for h in hits}
        gold = set(record["relevant_chunk_ids"])
        assert gold & got_ids, (
            f"{record['id']}: gold chunk(s) {gold!r} not in top-{settings.rerank_k}; "
            f"retriever returned {got_ids!r}"
        )
