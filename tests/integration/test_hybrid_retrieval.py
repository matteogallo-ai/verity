"""Integration: HybridRetriever against a live pgvector database.

Ingests the shipped corpus with the real LocalEmbedder + real ingestion pipeline,
then asserts that each answerable question's gold chunk lands in the top-rerank_k of
the retriever. Gold is resolved from the evidence against the chunks ingested here, as
the eval does: this machine's parse of a PDF may chunk it differently from the
labelling machine's. This is the pipeline-level sanity that CI exercises on every push.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.integration.db import database_url
from verity.config import get_settings
from verity.db import apply_pending
from verity.eval.labels import resolve_gold
from verity.eval.retrieval_run import load_full_dataset
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
from verity.types import Chunk

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = REPO_ROOT / "datasets" / "corpus"
DATASET_PATH = REPO_ROOT / "datasets" / "eval" / "questions.jsonl"


@pytest.fixture(scope="module")
def seeded_store() -> tuple[PgVectorStore, list[Chunk]]:
    url = database_url()
    apply_pending(url)
    store = PgVectorStore(database_url=url)

    import asyncio

    async def seed() -> list[Chunk]:
        embedder = LocalEmbedder()
        parser = create_default_parser()
        chunker = StructureAwareChunker()
        pipeline = IngestionPipeline(parser=parser, chunker=chunker, embedder=embedder, sink=store)
        result = await pipeline.ingest(discover_sources(str(CORPUS_DIR)))
        assert result.failures == []
        return result.chunks

    return store, asyncio.run(seed())


@pytest.mark.asyncio
async def test_gold_chunk_in_top_rerank_k(seeded_store: tuple[PgVectorStore, list[Chunk]]) -> None:
    store, chunks = seeded_store
    examples, problems = resolve_gold(load_full_dataset(DATASET_PATH), chunks)
    assert problems == []
    retriever = HybridRetriever(
        store=store,
        embedder=LocalEmbedder(),
        reranker=CrossEncoderReranker(),
    )
    settings = get_settings()
    for example in (e for e in examples if e.answerable and e.relevant_chunk_ids):
        hits = await retriever.retrieve(example.question, k=settings.rerank_k)
        got_ids = {str(h.chunk.id) for h in hits}
        gold = {str(cid) for cid in example.relevant_chunk_ids}
        assert gold & got_ids, (
            f"{example.id}: gold chunk(s) {gold!r} not in top-{settings.rerank_k}; "
            f"retriever returned {got_ids!r}"
        )
