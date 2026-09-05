"""Evaluation harness contracts — the differentiator.

The harness runs the *whole* agent over a labeled dataset and produces a
:class:`Scorecard`: retrieval precision/recall/nDCG, answer faithfulness, citation
accuracy, hallucination rate, refusal calibration, latency percentiles, and cost/query.
Runs are persisted with their git SHA so quality changes across commits are visible
(regression tracking), exactly as a production AI team tracks its models.

Design rule for CI reproducibility: retrieval and refusal metrics are deterministic and
run with local embeddings + a stubbed judge, so the eval suite is free and gate-able in
CI. The published headline scorecard comes from a separate real run (real judge model),
never fabricated.

Implementations land in S4.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from verity.types import (
    Answer,
    AnswerMetrics,
    EvalExample,
    EvalRun,
    RetrievalHit,
    RetrievalMetrics,
    Scorecard,
)


@runtime_checkable
class RetrievalMetric(Protocol):
    """Scores a retrieved set against gold relevant-chunk labels for one example."""

    def score(self, example: EvalExample, hits: list[RetrievalHit], k: int) -> RetrievalMetrics: ...


@runtime_checkable
class FaithfulnessJudge(Protocol):
    """LLM-as-judge: decomposes an answer into atomic claims and checks each against the
    retrieved context. Feeds faithfulness, citation accuracy, and hallucination rate.

    In CI a deterministic stub replaces the model so runs are free and reproducible.
    """

    async def judge(self, example: EvalExample, answer: Answer) -> AnswerMetrics: ...


@runtime_checkable
class Evaluator(Protocol):
    """Runs the full harness and produces one :class:`Scorecard`."""

    async def evaluate(self, dataset: list[EvalExample], git_sha: str) -> Scorecard: ...


@runtime_checkable
class RunStore(Protocol):
    """Persists and loads eval runs for regression tracking across commits."""

    def save(self, run: EvalRun) -> None: ...

    def history(self, dataset: str, limit: int = 50) -> list[EvalRun]: ...

    def latest(self, dataset: str) -> EvalRun | None: ...
