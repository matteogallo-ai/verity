"""Integration: verity eval run --judge stub end-to-end on pgvector.

Runs the full harness with a real LocalEmbedder + real cross-encoder + real
pgvector store, but with a scripted stub LLM (agent side) + mechanical stub judge.
Asserts the Scorecard has all four measurement tracks populated and that refusal
recall is a real number computed over the 5 out-of-scope questions.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from verity.agent.agent import create_default_agent
from verity.agent.scripted_stub import build_stub_router
from verity.config import get_settings
from verity.db import apply_pending
from verity.eval import (
    STUB_JUDGE_MODEL,
    AgentEvaluator,
    StubFaithfulnessJudge,
    load_full_dataset,
)
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
DATASET_PATH = REPO_ROOT / "datasets" / "eval" / "questions.jsonl"


def _database_url() -> str:
    return os.environ.get("VERITY_DATABASE_URL", get_settings().database_url)


@pytest.fixture(scope="module")
def seeded_retriever() -> HybridRetriever:
    url = _database_url()
    apply_pending(url)
    store = PgVectorStore(database_url=url)
    embedder = LocalEmbedder()

    import asyncio

    async def seed() -> None:
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
async def test_eval_harness_produces_full_scorecard(seeded_retriever: HybridRetriever) -> None:
    agent = create_default_agent(retriever=seeded_retriever, client=build_stub_router())
    judge = StubFaithfulnessJudge()

    examples = load_full_dataset(DATASET_PATH)
    assert len(examples) == 16

    evaluator = AgentEvaluator(
        agent=agent,
        judge=judge,
        embedding_model=get_settings().embedding_model,
        judge_model=STUB_JUDGE_MODEL,
        agent_model="stub-agent",
        dataset_name="questions",
    )
    result = await evaluator.run(examples, git_sha="int-test")
    sc = result.scorecard

    assert sc.n_examples == 16
    assert result.judge_model == STUB_JUDGE_MODEL
    assert result.agent_model == "stub-agent"

    # Every metric is populated within valid bounds.
    for value in (
        sc.retrieval.precision_at_k,
        sc.retrieval.recall_at_k,
        sc.retrieval.ndcg_at_k,
        sc.answer.faithfulness,
        sc.answer.citation_accuracy,
        sc.answer.hallucination_rate,
        sc.answer.refusal_precision,
        sc.answer.refusal_recall,
    ):
        assert 0.0 <= value <= 1.0
    assert sc.latency.p50_ms >= 0.0
    assert sc.cost_per_query_usd >= 0.0

    # Refusal is measured on all 5 out-of-scope.
    out_of_scope = [p for p in result.per_refusal if not p.expected_answerable]
    assert len(out_of_scope) == 5
    # And the near-miss questions are represented explicitly.
    ids = {p.example_id for p in out_of_scope}
    assert {"q-014", "q-015", "q-016"}.issubset(ids)
