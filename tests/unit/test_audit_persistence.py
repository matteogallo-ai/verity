"""Auditability contract for every persisted eval run.

The S7 pre-tag audit exposed this hole: the aggregate scorecard was written
to disk but the per-example answer text, citations, and judge verdict were
not. That meant a run of record could not be re-read after the fact —
a question like "what did the agent say for q-015?" had no answer without
re-spending on the judge.

These tests LOCK the persistence contract described in
:class:`verity.types.PerExampleAudit`:

- **Non-refused examples MUST carry non-empty ``answer_text`` and preserve the
  citations the synthesizer emitted.** If synthesis produced 3 citations, the
  audit record must persist all 3 (chunk_id + document_id + verbatim quote +
  char offsets).
- **Refused examples MUST carry a non-empty ``refusal_rationale``** — the
  operator reading the audit needs to know WHY the refusal happened.
- **``judge_claims`` is ``None`` iff the judge was bypassed (refusal path);
  otherwise it is a (possibly empty) tuple** carrying the per-claim
  ``ClaimVerdict``s so ``citation_accuracy`` and ``hallucination_rate`` can
  be traced back to specific claims without re-running the judge.
- **The full round-trip through ``FileRunStore`` is loss-free**: what the
  harness produces is what a later reader retrieves.

Zero LLM cost — driven by the stub agent + stub judge over the real 16-
question dataset.
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
    FileRunStore,
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


async def _run_stub_harness() -> tuple[FileRunStore, list, Path]:  # type: ignore[type-arg]
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
    result = await evaluator.run(examples, git_sha="audit_test")
    return result, examples  # type: ignore[return-value]


# --------------------------------------------------------------------------------------
# In-memory contract on HarnessResult / PerExampleAudit before persistence.
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_harness_produces_one_audit_record_per_dataset_example() -> None:
    """Baseline: the harness must record one :class:`PerExampleAudit` per
    dataset example, in the same order as the input dataset."""
    result, examples = await _run_stub_harness()
    assert len(result.per_audit) == len(examples)
    for audit, example in zip(result.per_audit, examples, strict=True):
        assert audit.example_id == example.id
        assert audit.question == example.question
        assert audit.expected_answerable == example.answerable


@pytest.mark.asyncio
async def test_non_refused_examples_carry_answer_text_and_citations() -> None:
    """Every non-refused example must have a non-empty answer text. If the
    synthesizer emitted citations, the audit record preserves them verbatim
    (quote + char offsets + chunk_id + document_id)."""
    result, _ = await _run_stub_harness()
    non_refused = [a for a in result.per_audit if not a.refused]
    assert non_refused, "stub agent should answer at least some questions — regression guard"
    for audit in non_refused:
        assert audit.answer_text.strip(), (
            f"non-refused audit for {audit.example_id} has empty answer_text"
        )
        for cit in audit.citations:
            assert cit.quote.strip(), f"empty quote in citation for {audit.example_id}"
            assert cit.char_end > cit.char_start, (
                f"invalid char span in citation for {audit.example_id}"
            )
            assert cit.chunk_id is not None
            assert cit.document_id is not None


@pytest.mark.asyncio
async def test_refused_examples_carry_a_non_empty_rationale() -> None:
    """A refusal without a rationale is un-auditable. Contract: every refused
    example must persist WHY it refused."""
    result, _ = await _run_stub_harness()
    refused = [a for a in result.per_audit if a.refused]
    assert refused, "stub agent must refuse at least some questions — regression guard"
    for audit in refused:
        assert audit.refusal_rationale.strip(), (
            f"refused audit for {audit.example_id} has empty refusal_rationale"
        )


@pytest.mark.asyncio
async def test_judge_claims_bypassed_on_refusals_populated_otherwise() -> None:
    """``judge_claims`` MUST be ``None`` iff the judge was bypassed (refusal
    short-circuit); otherwise it must be a tuple (possibly empty when the
    answer had zero atomic claims). Distinguishes "we skipped the judge for a
    refusal" from "the judge ran and had nothing to score"."""
    result, _ = await _run_stub_harness()
    for audit in result.per_audit:
        if audit.refused:
            assert audit.judge_claims is None, (
                f"refused audit for {audit.example_id} must carry judge_claims=None "
                f"(judge is bypassed on the refusal path)"
            )
        else:
            assert audit.judge_claims is not None, (
                f"non-refused audit for {audit.example_id} must carry a judge_claims tuple"
            )


@pytest.mark.asyncio
async def test_hits_used_persisted_for_every_example() -> None:
    """The retrieved chunks the agent had at its disposal are part of the
    audit record — a reader can inspect what evidence the agent saw without
    re-running retrieval."""
    result, _ = await _run_stub_harness()
    for audit in result.per_audit:
        # The stub agent + real retriever always retrieve at least one hit.
        assert len(audit.hits_used) > 0, f"no hits recorded for {audit.example_id}"


# --------------------------------------------------------------------------------------
# End-to-end round-trip through FileRunStore.
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_store_round_trip_preserves_every_audit_field(tmp_path: Path) -> None:
    """Persist → load → assert every audit record's fields survive intact.

    This is the load-bearing test: it guarantees a v1.0.0 run of record can be
    RE-READ tomorrow, next week, or by another engineer, without re-running
    the judge."""
    result, _ = await _run_stub_harness()
    run = result.as_eval_run()

    store = FileRunStore(tmp_path)
    store.save(run)

    reloaded = store.latest("questions")
    assert reloaded is not None
    assert len(reloaded.per_audit) == len(run.per_audit)
    for original, back in zip(run.per_audit, reloaded.per_audit, strict=True):
        assert back.example_id == original.example_id
        assert back.question == original.question
        assert back.expected_answerable == original.expected_answerable
        assert back.answer_text == original.answer_text
        assert back.refused == original.refused
        assert back.refusal_rationale == original.refusal_rationale
        assert back.citations == original.citations
        assert back.hits_used == original.hits_used
        assert back.judge_claims == original.judge_claims
        assert back.answer_metrics == original.answer_metrics
        assert back.usage == original.usage


def test_pre_v1_persisted_runs_load_without_per_audit(tmp_path: Path) -> None:
    """Backwards compat: a persisted run WITHOUT a ``per_audit`` key must load
    cleanly with an empty ``per_audit`` tuple. Guards against a future audit-
    field addition breaking pre-v1.0.0 archived runs."""
    # Shape mirrors the exact v0.7.x persisted format (no per_audit key).
    import json
    from datetime import UTC, datetime

    legacy = {
        "embedding_model": "bge-small",
        "judge_model": "stub-judge-v1",
        "agent_model": "stub-agent",
        "notes": None,
        "scorecard": {
            "git_sha": "legacy0",
            "created_at": datetime(2026, 8, 1, tzinfo=UTC).isoformat(),
            "dataset": "questions",
            "n_examples": 16,
            "retrieval": {
                "precision_at_k": 0.15,
                "recall_at_k": 1.0,
                "ndcg_at_k": 0.9,
                "k": 8,
            },
            "answer": {
                "faithfulness": 1.0,
                "citation_accuracy": 1.0,
                "hallucination_rate": 0.0,
                "refusal_precision": 1.0,
                "refusal_recall": 1.0,
            },
            "latency": {"p50_ms": 80.0, "p95_ms": 200.0, "p99_ms": 400.0},
            "cost_per_query_usd": 0.0,
            "cold_start_ms": None,
        },
    }
    (tmp_path / "eval_legacy0_questions.json").write_text(json.dumps(legacy), encoding="utf-8")
    store = FileRunStore(tmp_path)
    loaded = store.latest("questions")
    assert loaded is not None
    assert loaded.per_audit == ()
    # And a legacy run round-trips (write → read) to the same shape without
    # sprouting an empty ``per_audit`` key in the on-disk payload.
    store.save(loaded)
    payload = json.loads((tmp_path / "eval_legacy0_questions.json").read_text(encoding="utf-8"))
    assert "per_audit" not in payload
