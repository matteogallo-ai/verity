"""Evaluation package. Retrieval metrics + runner in S2; full harness in S4."""

from __future__ import annotations

from verity.eval.retrieval_metric import (
    BinaryRetrievalMetric,
    PerExampleScore,
    macro_average,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from verity.eval.retrieval_run import (
    RetrievalScorecard,
    current_git_sha,
    load_dataset,
    persist_scorecard,
    run_retrieval_eval,
)

__all__ = [
    "BinaryRetrievalMetric",
    "PerExampleScore",
    "RetrievalScorecard",
    "current_git_sha",
    "load_dataset",
    "macro_average",
    "ndcg_at_k",
    "persist_scorecard",
    "precision_at_k",
    "recall_at_k",
    "run_retrieval_eval",
]
