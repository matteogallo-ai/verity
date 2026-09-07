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
    PerExampleAudit,
    Provider,
    RetrievalMetrics,
    Scorecard,
)

log = structlog.get_logger(__name__)


class BudgetExceededError(RuntimeError):
    """Raised when a ``--judge live`` run's cumulative cost crosses the budget cap.

    Aborts the harness mid-loop so the operator does not accidentally spend
    beyond the pre-agreed ceiling. The ``verity eval run`` CLI surfaces this
    as a distinct exit code so scripts can distinguish "cost cap hit" from
    "metric floor missed"."""


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
class JudgeCallStamp:
    """Per-call judge provenance — what the router actually served this call.

    Both fields are ``None`` on the refusal short-circuit (the judge is
    bypassed and no LLM call happens). This lets the headline aggregator show
    e.g. "judge=claude-sonnet-4-6 (11/16), skipped for 5 refusals" instead of
    a boot-time default that lies about what ran.
    """

    example_id: str
    provider: Provider | None
    model: str | None


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
    # Per-call judge cost accumulator — the harness sums it across the loop so
    # ``total_cost_usd = sum(a.usage.cost_usd for a in answers) + judge_cost_usd``
    # is exact. Zero in ``--judge stub`` mode.
    judge_cost_usd: float = 0.0
    # Per-call judge provenance stamps — one entry per dataset example. Used by
    # ``build_headline_dict`` to emit a truthful aggregate ("who actually
    # served each judge call") rather than a boot-time model_label.
    judge_stamps: list[JudgeCallStamp] = field(default_factory=list)
    # Per-example audit records — the layer that lets a run of record be
    # RE-READ later without re-spending on the judge. Serialised verbatim
    # into ``EvalRun.per_audit`` by :meth:`as_eval_run`.
    per_audit: list[PerExampleAudit] = field(default_factory=list)

    def as_eval_run(self, *, notes: str | None = None) -> EvalRun:
        return EvalRun(
            scorecard=self.scorecard,
            embedding_model=self.embedding_model,
            judge_model=self.judge_model,
            agent_model=self.agent_model,
            notes=notes,
            per_audit=tuple(self.per_audit),
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

    async def run(
        self,
        dataset: list[EvalExample],
        *,
        git_sha: str,
        budget_usd_cap: float | None = None,
    ) -> HarnessResult:
        """Run the full harness. When ``budget_usd_cap`` is set, the loop aborts
        (raises :class:`BudgetExceededError`) the first time cumulative agent +
        judge cost crosses the cap. Intended for the one-shot live run: it is
        the hard safety guard that prevents an accidental $50 spend if the
        cost estimation was off."""
        answers: list[Answer] = []
        per_retrieval: list[PerExampleScore] = []
        per_answer: list[PerExampleAnswerMetrics] = []
        per_refusal: list[PerExampleRefusal] = []
        judge_stamps: list[JudgeCallStamp] = []
        per_audit: list[PerExampleAudit] = []
        retrieval_metric = BinaryRetrievalMetric()
        judge_cost_usd = 0.0

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

            # Accumulate judge cost from ``judge.last_usage`` — set by both
            # LLMFaithfulnessJudge and StubFaithfulnessJudge on every call
            # (zero for the stub, real for the LLM judge).
            judge_last_usage = getattr(self._judge, "last_usage", None)
            if judge_last_usage is not None:
                judge_cost_usd += float(judge_last_usage.cost_usd)

            # Per-call judge provenance — both StubFaithfulnessJudge and
            # LLMFaithfulnessJudge set ``last_provider`` / ``last_model`` per
            # call (both None on the refusal skip path, honest that no LLM
            # was called).
            judge_stamps.append(
                JudgeCallStamp(
                    example_id=example.id,
                    provider=getattr(self._judge, "last_provider", None),
                    model=getattr(self._judge, "last_model", None),
                )
            )

            per_refusal.append(
                PerExampleRefusal(
                    example_id=example.id,
                    expected_answerable=example.answerable,
                    observed_refused=answer.confidence.refused,
                )
            )

            # Audit record — the layer that makes this run RE-READABLE later
            # without re-spending on the judge. See ``PerExampleAudit`` for the
            # contract asserted by ``tests/unit/test_audit_persistence.py``.
            raw_judge_claims = getattr(self._judge, "last_claims", None)
            per_audit.append(
                PerExampleAudit(
                    example_id=example.id,
                    question=example.question,
                    expected_answerable=example.answerable,
                    answer_text=answer.text,
                    refused=answer.confidence.refused,
                    refusal_rationale=(
                        answer.confidence.rationale if answer.confidence.refused else ""
                    ),
                    citations=answer.citations,
                    hits_used=answer.hits_used,
                    judge_claims=(
                        tuple(raw_judge_claims) if isinstance(raw_judge_claims, tuple) else None
                    ),
                    answer_metrics=answer_metrics,
                    usage=answer.usage,
                )
            )

            if budget_usd_cap is not None:
                agent_cost_so_far = sum(a.usage.cost_usd for a in answers)
                cumulative = agent_cost_so_far + judge_cost_usd
                if cumulative > budget_usd_cap:
                    raise BudgetExceededError(
                        f"Aborted after {example.id}: cumulative cost "
                        f"${cumulative:.4f} exceeded budget cap ${budget_usd_cap:.2f}"
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
            judge_cost_usd=round(judge_cost_usd, 6),
            judge_stamps=judge_stamps,
            per_audit=per_audit,
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
    "BudgetExceededError",
    "HarnessResult",
    "JudgeCallStamp",
    "PerExampleAnswerMetrics",
    "PerExampleRefusal",
]
