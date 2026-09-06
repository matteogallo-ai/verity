"""End-to-end harness with stub agent + stub judge over the real 16-question dataset.

No network, no keys, no DB — the in-memory store + scripted stub + mechanical judge
run entirely locally, so this test replaces the integration lane for reviewers who
can't provision Postgres.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from verity.agent.agent import create_default_agent
from verity.agent.scripted_stub import build_stub_router
from verity.eval import (
    STUB_JUDGE_MODEL,
    AgentEvaluator,
    StubFaithfulnessJudge,
    load_full_dataset,
)
from verity.ingestion.chunker import StructureAwareChunker
from verity.ingestion.embedder import LocalEmbedder
from verity.ingestion.parsers import create_default_parser
from verity.ingestion.pipeline import IngestionPipeline, discover_sources
from verity.retrieval import CrossEncoderReranker, HybridRetriever, InMemoryVectorStore

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS = REPO_ROOT / "datasets" / "corpus"
DATASET = REPO_ROOT / "datasets" / "eval" / "questions.jsonl"


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


async def _seed_retriever() -> HybridRetriever:
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
    return HybridRetriever(
        store=store,
        embedder=embedder,
        reranker=CrossEncoderReranker(model="fake", backend=_FakeRerankerBackend()),
    )


@pytest.mark.asyncio
async def test_harness_produces_full_scorecard_with_stub_labels() -> None:
    retriever = await _seed_retriever()
    agent = create_default_agent(retriever=retriever, client=build_stub_router())
    judge = StubFaithfulnessJudge()

    examples = load_full_dataset(DATASET)
    assert len(examples) == 16, f"expected 16-question dataset, got {len(examples)}"

    evaluator = AgentEvaluator(
        agent=agent,
        judge=judge,
        embedding_model="fake",
        judge_model=STUB_JUDGE_MODEL,
        agent_model="stub-agent",
        dataset_name="questions",
    )
    result = await evaluator.run(examples, git_sha="test0000")

    # Scorecard shape.
    sc = result.scorecard
    assert sc.n_examples == 16
    assert sc.git_sha == "test0000"
    assert 0.0 <= sc.retrieval.precision_at_k <= 1.0
    assert 0.0 <= sc.retrieval.recall_at_k <= 1.0
    assert 0.0 <= sc.retrieval.ndcg_at_k <= 1.0
    # Refusal is REAL — not stub. Computed from 11 answerable + 5 out-of-scope.
    assert 0.0 <= sc.answer.refusal_recall <= 1.0
    assert 0.0 <= sc.answer.refusal_precision <= 1.0
    # The stub judge AND stub agent carried through — traceability triple.
    assert result.judge_model == STUB_JUDGE_MODEL
    assert result.agent_model == "stub-agent"
    # And the EvalRun round-trip preserves it.
    run = result.as_eval_run()
    assert run.agent_model == "stub-agent"

    # Every example got a refusal outcome recorded.
    assert len(result.per_refusal) == 16
    out_of_scope_ids = {"q-004", "q-006", "q-014", "q-015", "q-016"}
    observed_out_of_scope = {p.example_id for p in result.per_refusal if not p.expected_answerable}
    assert observed_out_of_scope == out_of_scope_ids


@pytest.mark.asyncio
async def test_harness_refusal_recall_captures_hallucinations_on_near_miss() -> None:
    """The near-miss questions q-014/q-015/q-016 test whether the agent invents
    when there's *plausible-looking* adjacent context. The scripted agent stub
    only answers when a theme matches, so all 5 out-of-scope should be refused
    → refusal_recall == 1.0 in this configuration. If a future refactor causes
    the stub to hallucinate on a near-miss, this assertion will catch it."""
    retriever = await _seed_retriever()
    agent = create_default_agent(retriever=retriever, client=build_stub_router())
    judge = StubFaithfulnessJudge()

    examples = load_full_dataset(DATASET)
    evaluator = AgentEvaluator(
        agent=agent,
        judge=judge,
        embedding_model="fake",
        judge_model=STUB_JUDGE_MODEL,
        agent_model="stub-agent",
        dataset_name="questions",
    )
    result = await evaluator.run(examples, git_sha="test0000")

    # Every out-of-scope must have been refused.
    for p in result.per_refusal:
        if not p.expected_answerable:
            assert p.observed_refused, f"stub agent hallucinated on out-of-scope {p.example_id}"
    assert result.scorecard.answer.refusal_recall == 1.0
