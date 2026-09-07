"""End-to-end evaluation harness that produces a full :class:`Scorecard`.

Runs the agent on every question in the labeled dataset and merges four
independently-testable measurement tracks into one artefact:

1. **Retrieval** (deterministic) — precision@k, recall@k, nDCG@k on the
   answerable-and-labeled subset. Reuses :mod:`verity.eval.retrieval_metric`.
2. **Answer quality** (via injected :class:`FaithfulnessJudge`) — faithfulness,
   citation_accuracy, hallucination_rate. In ``--judge stub`` these come from the
   scripted mechanical judge and the Scorecard's ``judge_model`` records that
   explicitly (``stub-judge-v1``).
3. **Refusal calibration** (deterministic) — refusal_precision + refusal_recall
   from the (expected_answerable, observed_refused) pairs across the whole
   dataset, including the near-miss out-of-scope questions.
4. **Operational** — latency percentiles from ``Answer.usage.latency_ms``,
   cost/query mean from ``Answer.usage.cost_usd`` (0.0 in stub mode).

The Scorecard is bound to a git SHA + the embedding model + the judge model, so
runs are comparable across commits — that is how ``verity eval compare`` surfaces
regressions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import cast

import structlog

from verity.agent.base import Agent
from verity.config import get_settings
from verity.eval.base import Evaluator, FaithfulnessJudge
from verity.eval.refusal import RefusalOutcome, build_confusion
from verity.eval.retrieval_metric import (
    BinaryRetrievalMetric,
    PerExampleScore,
    macro_average,
)
from verity.types import (
    Answer,
    AnswerMetrics,
    EvalExample,
    EvalRun,
    LatencyMetrics,
    RetrievalMetrics,
    Scorecard,
)

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class PerExampleAnswerMetrics:
    example_id: str
    metrics: AnswerMetrics


@dataclass(frozen=True)
class PerExampleRefusal:
    example_id: str
    expected_answerable: bool
    observed_refused: bool


@dataclass(frozen=True)
class HarnessResult:
    """Full per-example breakdown alongside the aggregated :class:`Scorecard`.

    Persisted verbatim in the ``EvalRun`` record so ``verity eval compare`` can
    surface per-example regressions later without re-running the harness.
    """

    scorecard: Scorecard
    embedding_model: str
    judge_model: str
    agent_model: str
    answers: list[Answer] = field(default_factory=list)
    per_retrieval: list[PerExampleScore] = field(default_factory=list)
    per_answer: list[PerExampleAnswerMetrics] = field(default_factory=list)
    per_refusal: list[PerExampleRefusal] = field(default_factory=list)

    def as_eval_run(self, *, notes: str | None = None) -> EvalRun:
        return EvalRun(
            scorecard=self.scorecard,
            embedding_model=self.embedding_model,
            judge_model=self.judge_model,
            agent_model=self.agent_model,
            notes=notes,
        )


class AgentEvaluator(Evaluator):
    """Runs an :class:`Agent` over a dataset and aggregates a :class:`Scorecard`."""

    def __init__(
        self,
        *,
        agent: Agent,
        judge: FaithfulnessJudge,
        embedding_model: str,
        judge_model: str,
        agent_model: str,
        dataset_name: str,
        k: int | None = None,
        warmup: bool = True,
    ) -> None:
        settings = get_settings()
        self._agent = agent
        self._judge = judge
        self._embedding_model = embedding_model
        self._judge_model = judge_model
        self._agent_model = agent_model
        self._dataset_name = dataset_name
        self._k = k if k is not None else settings.rerank_k
        self._warmup = warmup

    async def evaluate(self, dataset: list[EvalExample], git_sha: str) -> Scorecard:
        result = await self.run(dataset, git_sha=git_sha)
        return result.scorecard

    async def run(self, dataset: list[EvalExample], *, git_sha: str) -> HarnessResult:
        answers: list[Answer] = []
        per_retrieval: list[PerExampleScore] = []
        per_answer: list[PerExampleAnswerMetrics] = []
        per_refusal: list[PerExampleRefusal] = []
        retrieval_metric = BinaryRetrievalMetric()

        cold_start_ms: float | None = None
        if self._warmup:
            # One throwaway agent call primes every lazy component (embedder,
            # reranker, tokenisers) so the timed loop below reflects
            # steady-state serving latency, not first-boot model loads.
            from verity.observability.warmup import agent_warmup

            cold_start_ms = await agent_warmup(self._agent)
            log.info("harness_warmup_done", cold_start_ms=cold_start_ms)

        for example in dataset:
            answer = await self._agent.answer(example.question)
            answers.append(answer)
            log.info(
                "harness_answered",
                example_id=example.id,
                refused=answer.confidence.refused,
                n_citations=len(answer.citations),
                elapsed_ms=answer.usage.latency_ms,
            )

            # Retrieval: only meaningful on labeled answerable questions.
            if example.answerable and example.relevant_chunk_ids:
                per_retrieval.append(
                    PerExampleScore(
                        example_id=example.id,
                        metrics=retrieval_metric.score(example, list(answer.hits_used), k=self._k),
                    )
                )

            # Answer quality: judge every example (refusals get a well-defined
            # convention in the judge — see judge.py docstring).
            answer_metrics = await self._judge.judge(example, answer)
            per_answer.append(
                PerExampleAnswerMetrics(example_id=example.id, metrics=answer_metrics)
            )

            per_refusal.append(
                PerExampleRefusal(
                    example_id=example.id,
                    expected_answerable=example.answerable,
                    observed_refused=answer.confidence.refused,
                )
            )

        retrieval_agg = _aggregate_retrieval(per_retrieval, k=self._k)
        answer_agg = _aggregate_answer(per_answer, per_refusal)
        latency = _latency_percentiles(answers)
        cost_per_query = _mean_cost(answers)

        scorecard = Scorecard(
            git_sha=git_sha,
            created_at=datetime.now(UTC),
            dataset=self._dataset_name,
            n_examples=len(dataset),
            retrieval=retrieval_agg,
            answer=answer_agg,
            latency=latency,
            cost_per_query_usd=cost_per_query,
            cold_start_ms=cold_start_ms,
        )
        return HarnessResult(
            scorecard=scorecard,
            embedding_model=self._embedding_model,
            judge_model=self._judge_model,
            agent_model=self._agent_model,
            answers=answers,
            per_retrieval=per_retrieval,
            per_answer=per_answer,
            per_refusal=per_refusal,
        )


def _aggregate_retrieval(per_example: list[PerExampleScore], *, k: int) -> RetrievalMetrics:
    return macro_average(per_example, k=k)


def _aggregate_answer(
    per_answer: list[PerExampleAnswerMetrics],
    per_refusal: list[PerExampleRefusal],
) -> AnswerMetrics:
    """Mean faithfulness / citation_accuracy / hallucination_rate + real refusal metrics."""
    if per_answer:
        n = float(len(per_answer))
        faithfulness = sum(p.metrics.faithfulness for p in per_answer) / n
        citation_accuracy = sum(p.metrics.citation_accuracy for p in per_answer) / n
        hallucination_rate = sum(p.metrics.hallucination_rate for p in per_answer) / n
    else:
        faithfulness = citation_accuracy = hallucination_rate = 0.0

    outcomes = [
        RefusalOutcome(
            example_id=p.example_id,
            expected_answerable=p.expected_answerable,
            observed_refused=p.observed_refused,
        )
        for p in per_refusal
    ]
    confusion = build_confusion(outcomes)

    return AnswerMetrics(
        faithfulness=faithfulness,
        citation_accuracy=citation_accuracy,
        hallucination_rate=hallucination_rate,
        refusal_precision=confusion.refusal_precision,
        refusal_recall=confusion.refusal_recall,
    )


def _latency_percentiles(answers: list[Answer]) -> LatencyMetrics:
    if not answers:
        return LatencyMetrics(p50_ms=0.0, p95_ms=0.0, p99_ms=0.0)
    values = sorted(a.usage.latency_ms for a in answers)
    return LatencyMetrics(
        p50_ms=_percentile(values, 50),
        p95_ms=_percentile(values, 95),
        p99_ms=_percentile(values, 99),
    )


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    # Linear-interpolation percentile (numpy default), computed by hand to avoid a
    # numpy import in the harness. Rank in [0, n-1].
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    frac = rank - low
    return float(sorted_values[low] + (sorted_values[high] - sorted_values[low]) * frac)


def _mean_cost(answers: list[Answer]) -> float:
    if not answers:
        return 0.0
    total = sum(a.usage.cost_usd for a in answers)
    return float(total) / len(answers)


# Silence a stray import warning if AnswerMetrics ever changes shape.
_ = cast


__all__ = [
    "AgentEvaluator",
    "HarnessResult",
    "PerExampleAnswerMetrics",
    "PerExampleRefusal",
]
