"""Retrieval metrics: precision@k, recall@k, nDCG@k — hand-computed cases."""

from __future__ import annotations

import math

import pytest

from verity.eval.retrieval_metric import (
    BinaryRetrievalMetric,
    PerExampleScore,
    macro_average,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from verity.types import (
    Chunk,
    DocumentId,
    EvalExample,
    RetrievalHit,
    RetrievalMetrics,
    RetrieverKind,
)


def test_precision_recall_ndcg_all_hits_correct() -> None:
    retrieved = ["a", "b", "c"]
    gold = {"a", "b", "c"}
    assert precision_at_k(retrieved, gold, k=3) == 1.0
    assert recall_at_k(retrieved, gold, k=3) == 1.0
    assert ndcg_at_k(retrieved, gold, k=3) == 1.0


def test_precision_recall_partial_and_ordering() -> None:
    retrieved = ["x", "a", "y", "b"]  # 2/4 relevant, 2/3 recall
    gold = {"a", "b", "c"}
    assert precision_at_k(retrieved, gold, k=4) == pytest.approx(0.5)
    assert recall_at_k(retrieved, gold, k=4) == pytest.approx(2.0 / 3.0)
    # DCG: rank 1 (a) → 1/log2(3), rank 3 (b) → 1/log2(5)
    dcg = 1.0 / math.log2(3) + 1.0 / math.log2(5)
    # IDCG: 3 gold, k=4 → top 3 slots ideal → 1 + 1/log2(3) + 1/log2(4)
    idcg = 1.0 + 1.0 / math.log2(3) + 1.0 / math.log2(4)
    assert ndcg_at_k(retrieved, gold, k=4) == pytest.approx(dcg / idcg)


def test_precision_empty_retrieval_is_zero() -> None:
    assert precision_at_k([], {"a"}, k=5) == 0.0


def test_recall_and_ndcg_raise_on_empty_gold() -> None:
    with pytest.raises(ValueError):
        recall_at_k(["a"], set(), k=5)
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], set(), k=5)


def test_precision_when_k_exceeds_hits() -> None:
    # 1 relevant in a 2-hit list, evaluate at k=5 → denom is min(k, len(hits))=2
    assert precision_at_k(["a", "x"], {"a"}, k=5) == 0.5


def test_ndcg_bounds_and_ideal_ordering() -> None:
    # Reversed vs ideal: relevant chunk at rank 1 rather than rank 0.
    retrieved = ["x", "a"]
    gold = {"a"}
    ndcg = ndcg_at_k(retrieved, gold, k=2)
    assert 0.0 < ndcg < 1.0
    # Ideal ordering scores exactly 1.
    assert ndcg_at_k(["a", "x"], {"a"}, k=2) == 1.0


def test_binary_metric_scores_one_example() -> None:
    gold_id = "3a3f2a58-2b28-5d34-9e6e-000000000001"
    other_id = "3a3f2a58-2b28-5d34-9e6e-000000000002"
    hits = [
        RetrievalHit(
            chunk=Chunk(
                id=other_id,  # type: ignore[arg-type]
                document_id=DocumentId("00000000-0000-0000-0000-000000000000"),
                text="wrong",
                ordinal=0,
                char_start=0,
                char_end=5,
            ),
            score=1.0,
            kind=RetrieverKind.RERANKED,
            rank=0,
        ),
        RetrievalHit(
            chunk=Chunk(
                id=gold_id,  # type: ignore[arg-type]
                document_id=DocumentId("00000000-0000-0000-0000-000000000000"),
                text="right",
                ordinal=1,
                char_start=6,
                char_end=11,
            ),
            score=0.9,
            kind=RetrieverKind.RERANKED,
            rank=1,
        ),
    ]
    example = EvalExample(
        id="q-x",
        question="q",
        answerable=True,
        reference_answer="r",
        relevant_chunk_ids=(gold_id,),  # type: ignore[arg-type]
    )
    result = BinaryRetrievalMetric().score(example, hits, k=2)
    assert isinstance(result, RetrievalMetrics)
    assert result.precision_at_k == 0.5
    assert result.recall_at_k == 1.0
    assert result.k == 2


def test_macro_average_is_mean_of_per_example() -> None:
    per = [
        PerExampleScore(
            example_id="q1",
            metrics=RetrievalMetrics(precision_at_k=1.0, recall_at_k=0.5, ndcg_at_k=0.9, k=8),
        ),
        PerExampleScore(
            example_id="q2",
            metrics=RetrievalMetrics(precision_at_k=0.0, recall_at_k=0.0, ndcg_at_k=0.0, k=8),
        ),
    ]
    agg = macro_average(per, k=8)
    assert agg.precision_at_k == 0.5
    assert agg.recall_at_k == 0.25
    assert agg.ndcg_at_k == pytest.approx(0.45)
    assert agg.k == 8


def test_macro_average_empty_returns_zeros() -> None:
    agg = macro_average([], k=8)
    assert agg.precision_at_k == 0.0
    assert agg.recall_at_k == 0.0
    assert agg.ndcg_at_k == 0.0
    assert agg.k == 8
