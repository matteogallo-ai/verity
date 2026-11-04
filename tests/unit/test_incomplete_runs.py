"""A paid run that stops early or loses an example still writes what it paid for.

An interrupted run (budget cap reached) or one with errored examples (agent or judge
outage) is persisted as an *incomplete* artefact: every paid example is in it, errors
included, under a name of its own that never overwrites a run of record and that
history, comparison and the headline ignore.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.unit.test_harness import DATASET, _seed_retriever
from verity.agent.agent import create_default_agent
from verity.agent.scripted_stub import build_stub_router
from verity.eval import AgentEvaluator, BudgetExceededError, load_full_dataset
from verity.eval.judge import STUB_JUDGE_MODEL, StubFaithfulnessJudge
from verity.eval.run_store import FileRunStore
from verity.llm.base import ProviderUnavailableError
from verity.types import Answer, AnswerError, ErrorKind, IncompleteReason, UsageStats


class _CostlyAgent:
    """The stub agent with every answer billed $1; ``failing`` questions error."""

    def __init__(self, inner: object, failing: frozenset[str]) -> None:
        self._inner = inner
        self._failing = failing

    async def answer(self, question: str) -> Answer:
        if question in self._failing:
            return Answer(
                query=question,
                text="",
                error=AnswerError(kind=ErrorKind.PROVIDER_UNAVAILABLE, detail="down"),
                usage=UsageStats(cost_usd=1.0, llm_calls=1),
                stage_ms={"agent.decompose": 1.0},
            )
        answer = await self._inner.answer(question)  # type: ignore[attr-defined]
        return answer.model_copy(
            update={"usage": answer.usage.model_copy(update={"cost_usd": 1.0})}
        )


class _FlakyJudge:
    def __init__(self, failing_ids: frozenset[str]) -> None:
        self._stub = StubFaithfulnessJudge()
        self._failing = failing_ids

    def __getattr__(self, name: str) -> object:
        return getattr(self._stub, name)

    async def judge(self, example, answer):  # type: ignore[no-untyped-def]
        if example.id in self._failing:
            raise ProviderUnavailableError("judge provider down")
        return await self._stub.judge(example, answer)


async def _evaluator(
    *, failing_questions: frozenset[str] = frozenset(), failing_judge: frozenset[str] = frozenset()
) -> AgentEvaluator:
    inner = create_default_agent(retriever=await _seed_retriever(), client=build_stub_router())
    return AgentEvaluator(
        agent=_CostlyAgent(inner, failing_questions),  # type: ignore[arg-type]
        judge=_FlakyJudge(failing_judge),  # type: ignore[arg-type]
        embedding_model="fake",
        judge_model=STUB_JUDGE_MODEL,
        agent_model="stub-agent",
        dataset_name="questions",
        warmup=False,
    )


@pytest.mark.asyncio
async def test_errored_examples_are_kept_with_what_they_cost() -> None:
    examples = load_full_dataset(DATASET)
    agent_down, judge_down = examples[0], examples[1]
    evaluator = await _evaluator(
        failing_questions=frozenset({agent_down.question}),
        failing_judge=frozenset({judge_down.id}),
    )
    result = await evaluator.run(examples, git_sha="t")

    assert result.scorecard.incomplete_reason is IncompleteReason.ERRORED_EXAMPLES
    errored = {e.example_id: e for e in result.errored}
    assert set(errored) == {agent_down.id, judge_down.id}
    assert errored[agent_down.id].failed_stage == "agent"
    assert errored[agent_down.id].error == "provider_unavailable"
    assert errored[agent_down.id].usage.cost_usd == 1.0
    assert errored[agent_down.id].stage_ms == {"agent.decompose": 1.0}
    # The judge failed after the answer was paid for: the answer is kept to rejudge.
    assert errored[judge_down.id].failed_stage == "judge"
    assert errored[judge_down.id].error == "ProviderUnavailableError"
    assert errored[judge_down.id].answer_text
    assert errored[judge_down.id].usage.cost_usd == 1.0
    paid = len(result.per_audit) + len(result.errored)
    assert paid == len(examples)


@pytest.mark.asyncio
async def test_a_clean_run_is_complete() -> None:
    result = await (await _evaluator()).run(load_full_dataset(DATASET), git_sha="t")
    assert result.scorecard.incomplete_reason is None
    assert result.errored == []


@pytest.mark.asyncio
async def test_a_budget_abort_carries_every_example_paid_so_far() -> None:
    examples = load_full_dataset(DATASET)
    evaluator = await _evaluator(failing_judge=frozenset({examples[1].id}))
    with pytest.raises(BudgetExceededError) as caught:
        await evaluator.run(examples, git_sha="t", budget_usd_cap=2.5)
    partial = caught.value.result
    assert partial.scorecard.incomplete_reason is IncompleteReason.BUDGET_EXCEEDED
    # $1 per answer and a $2.50 cap: the third paid example crosses it.
    assert len(partial.per_audit) + len(partial.errored) == 3
    assert [e.example_id for e in partial.errored] == [examples[1].id]


@pytest.mark.asyncio
async def test_an_incomplete_run_never_replaces_or_joins_the_runs_of_record(
    tmp_path: Path,
) -> None:
    examples = load_full_dataset(DATASET)
    complete = await (await _evaluator()).run(examples, git_sha="same")
    store = FileRunStore(tmp_path)
    record_path = store.save(complete.as_eval_run())
    record_bytes = record_path.read_bytes()

    flaky = await _evaluator(failing_judge=frozenset({examples[1].id}))
    first = store.save((await flaky.run(examples, git_sha="same")).as_eval_run())
    second = store.save((await flaky.run(examples, git_sha="same")).as_eval_run())

    assert record_path.read_bytes() == record_bytes
    assert len({record_path, first, second}) == 3
    assert ".incomplete-" in first.name and ".incomplete-" in second.name
    assert [r.scorecard.incomplete_reason for r in store.history("questions")] == [None]

    saved = json.loads(first.read_text(encoding="utf-8"))
    assert saved["scorecard"]["incomplete_reason"] == "errored_examples"
    assert [e["example_id"] for e in saved["errored"]] == [examples[1].id]
    assert saved["errored"][0]["answer_text"]
