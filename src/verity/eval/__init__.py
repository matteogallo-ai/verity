"""Evaluation package. S4 wires the full harness on top of the S2 retrieval metrics."""

from __future__ import annotations

from verity.eval.harness import (
    AgentEvaluator,
    HarnessResult,
    PerExampleAnswerMetrics,
    PerExampleRefusal,
)
from verity.eval.judge import (
    STUB_JUDGE_MODEL,
    LLMFaithfulnessJudge,
    StubFaithfulnessJudge,
)
from verity.eval.refusal import ConfusionMatrix, RefusalOutcome, build_confusion
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
    load_full_dataset,
    persist_scorecard,
    run_retrieval_eval,
)
from verity.eval.run_store import FileRunStore, MetricDelta, diff_runs

__all__ = [
    "STUB_JUDGE_MODEL",
    "AgentEvaluator",
    "BinaryRetrievalMetric",
    "ConfusionMatrix",
    "FileRunStore",
    "HarnessResult",
    "LLMFaithfulnessJudge",
    "MetricDelta",
    "PerExampleAnswerMetrics",
    "PerExampleRefusal",
    "PerExampleScore",
    "RefusalOutcome",
    "RetrievalScorecard",
    "StubFaithfulnessJudge",
    "build_confusion",
    "current_git_sha",
    "diff_runs",
    "load_dataset",
    "load_full_dataset",
    "macro_average",
    "ndcg_at_k",
    "persist_scorecard",
    "precision_at_k",
    "recall_at_k",
    "run_retrieval_eval",
]
