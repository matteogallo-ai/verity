"""End-to-end harness with stub agent + stub judge over the real 19-question dataset.

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
    assert any(e.answerable for e in examples) and any(not e.answerable for e in examples)

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
    assert sc.n_examples == len(examples)
    assert sc.git_sha == "test0000"
    assert 0.0 <= sc.retrieval.precision_at_k <= 1.0
    assert 0.0 <= sc.retrieval.recall_at_k <= 1.0
    assert 0.0 <= sc.retrieval.ndcg_at_k <= 1.0
    # Refusal is REAL — not stub. Computed from 11 answerable + 8 out-of-scope.
    assert 0.0 <= sc.answer.refusal_recall <= 1.0
    assert 0.0 <= sc.answer.refusal_precision <= 1.0
    # The stub judge AND stub agent carried through — traceability triple.
    assert result.judge_model == STUB_JUDGE_MODEL
    assert result.agent_model == "stub-agent"
    # And the EvalRun round-trip preserves it.
    run = result.as_eval_run()
    assert run.agent_model == "stub-agent"

    # Every example got a refusal outcome recorded.
    assert len(result.per_refusal) == len(examples)
    out_of_scope_ids = {e.id for e in examples if not e.answerable}
    observed_out_of_scope = {p.example_id for p in result.per_refusal if not p.expected_answerable}
    assert observed_out_of_scope == out_of_scope_ids


@pytest.mark.asyncio
async def test_harness_refusal_recall_captures_hallucinations_on_near_miss() -> None:
    """The near-miss questions q-014/q-015 test whether the agent invents when there's
    *plausible-looking* adjacent context. The scripted agent stub only answers when a
    theme matches; its themes are written for the demo questions q-001 to q-019, so on
    those every out-of-scope question is refused (refusal_recall == 1.0). The stub's
    keyword matching says nothing about the later questions, which only a live run
    measures."""
    retriever = await _seed_retriever()
    agent = create_default_agent(retriever=retriever, client=build_stub_router())
    judge = StubFaithfulnessJudge()

    examples = [e for e in load_full_dataset(DATASET) if e.id <= "q-019"]
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


@pytest.mark.asyncio
async def test_errored_examples_are_counted_apart_from_every_metric() -> None:
    """One example hits an outage: it is listed, not scored as a refusal or an answer."""
    from verity.types import Answer, AnswerError, ErrorKind

    retriever = await _seed_retriever()
    real = create_default_agent(retriever=retriever, client=build_stub_router())
    examples = load_full_dataset(DATASET)
    failing_question = examples[0].question

    class _FlakyAgent:
        async def answer(self, question: str) -> Answer:
            if question == failing_question:
                return Answer(
                    query=question,
                    text="",
                    error=AnswerError(kind=ErrorKind.PROVIDER_UNAVAILABLE, detail="down"),
                )
            return await real.answer(question)

    evaluator = AgentEvaluator(
        agent=_FlakyAgent(),  # type: ignore[arg-type]
        judge=StubFaithfulnessJudge(),
        embedding_model="fake",
        judge_model=STUB_JUDGE_MODEL,
        agent_model="stub-agent",
        dataset_name="questions",
    )
    result = await evaluator.run(examples, git_sha="test0000")

    assert result.scorecard.errored_example_ids == (examples[0].id,)
    assert len(result.per_refusal) == len(examples) - 1
    assert examples[0].id not in {r.example_id for r in result.per_refusal}


def _billed_outage_agent(real: object, failing_question: str, cost_usd: float) -> object:
    """An agent whose answer to ``failing_question`` errors after spending ``cost_usd``."""
    from verity.types import Answer, AnswerError, ErrorKind, UsageStats

    class _Agent:
        async def answer(self, question: str) -> Answer:
            if question == failing_question:
                return Answer(
                    query=question,
                    text="",
                    error=AnswerError(kind=ErrorKind.PROVIDER_UNAVAILABLE, detail="down"),
                    usage=UsageStats(input_tokens=400, llm_calls=2, cost_usd=cost_usd),
                )
            return await real.answer(question)  # type: ignore[attr-defined]

    return _Agent()


def _evaluator(agent: object) -> AgentEvaluator:
    return AgentEvaluator(
        agent=agent,  # type: ignore[arg-type]
        judge=StubFaithfulnessJudge(),
        embedding_model="fake",
        judge_model=STUB_JUDGE_MODEL,
        agent_model="stub-agent",
        dataset_name="questions",
    )


@pytest.mark.asyncio
async def test_cost_spent_on_an_errored_example_counts_in_the_run_cost() -> None:
    examples = load_full_dataset(DATASET)
    real = create_default_agent(retriever=await _seed_retriever(), client=build_stub_router())
    agent = _billed_outage_agent(real, examples[0].question, cost_usd=0.5)

    result = await _evaluator(agent).run(examples, git_sha="test0000")

    assert result.errored_cost_usd == pytest.approx(0.5)
    # The stub agent's answers cost nothing: the mean is the errored spend over every
    # attempted question.
    assert result.scorecard.cost_per_query_usd == pytest.approx(0.5 / len(examples))


@pytest.mark.asyncio
async def test_budget_cap_counts_the_cost_of_errored_examples() -> None:
    from verity.eval.harness import BudgetExceededError

    examples = load_full_dataset(DATASET)
    real = create_default_agent(retriever=await _seed_retriever(), client=build_stub_router())
    agent = _billed_outage_agent(real, examples[0].question, cost_usd=0.5)

    with pytest.raises(BudgetExceededError, match=examples[0].id):
        await _evaluator(agent).run(examples, git_sha="test0000", budget_usd_cap=0.1)


@pytest.mark.asyncio
async def test_a_judge_outage_errors_one_example_instead_of_losing_the_run() -> None:
    """The agent calls before a judge failure are paid for; an exception escaping the
    loop would discard every result of the run and invite a second spend."""
    from verity.llm.base import ProviderUnavailableError

    retriever = await _seed_retriever()
    agent = create_default_agent(retriever=retriever, client=build_stub_router())
    examples = load_full_dataset(DATASET)
    failing_id = examples[1].id
    stub = StubFaithfulnessJudge()

    class _FlakyJudge:
        def __getattr__(self, name: str) -> object:
            return getattr(stub, name)

        async def judge(self, example, answer):  # type: ignore[no-untyped-def]
            if example.id == failing_id:
                raise ProviderUnavailableError("judge provider down")
            return await stub.judge(example, answer)

    evaluator = AgentEvaluator(
        agent=agent,
        judge=_FlakyJudge(),  # type: ignore[arg-type]
        embedding_model="fake",
        judge_model=STUB_JUDGE_MODEL,
        agent_model="stub-agent",
        dataset_name="questions",
    )
    result = await evaluator.run(examples, git_sha="test0000")

    assert result.scorecard.errored_example_ids == (failing_id,)
    scored = {r.example_id for r in result.per_refusal}
    assert failing_id not in scored and len(scored) == len(examples) - 1
    assert failing_id not in {a.example_id for a in result.per_audit}
    assert failing_id not in {r.example_id for r in result.per_retrieval}


def _down_judge(exc: Exception) -> object:
    stub = StubFaithfulnessJudge()

    class _DownJudge:
        def __getattr__(self, name: str) -> object:
            return getattr(stub, name)

        async def judge(self, example, answer):  # type: ignore[no-untyped-def]
            raise exc

    return _DownJudge()


async def _costly_agent() -> object:
    """The stub agent, with every answer billed $1 so a cap can be crossed."""
    inner = create_default_agent(retriever=await _seed_retriever(), client=build_stub_router())

    class _CostlyAgent:
        async def answer(self, question: str):  # type: ignore[no-untyped-def]
            answer = await inner.answer(question)
            usage = answer.usage.model_copy(update={"cost_usd": 1.0})
            return answer.model_copy(update={"usage": usage})

    return _CostlyAgent()


@pytest.mark.asyncio
async def test_the_budget_cap_counts_answers_whose_judge_failed() -> None:
    """A judge that stays down must not let the agent spend past the cap."""
    from verity.eval import BudgetExceededError
    from verity.llm.base import ProviderUnavailableError

    evaluator = AgentEvaluator(
        agent=await _costly_agent(),  # type: ignore[arg-type]
        judge=_down_judge(ProviderUnavailableError("down")),  # type: ignore[arg-type]
        embedding_model="fake",
        judge_model=STUB_JUDGE_MODEL,
        agent_model="stub-agent",
        dataset_name="questions",
    )
    with pytest.raises(BudgetExceededError):
        await evaluator.run(load_full_dataset(DATASET), git_sha="test0000", budget_usd_cap=2.5)


@pytest.mark.asyncio
async def test_a_rejected_judge_key_stops_the_run() -> None:
    """A bad key fails every example the same way: carrying on would spend the agent
    on the whole dataset for no score."""
    from verity.llm.base import AuthenticationError

    evaluator = AgentEvaluator(
        agent=await _costly_agent(),  # type: ignore[arg-type]
        judge=_down_judge(AuthenticationError("bad key")),  # type: ignore[arg-type]
        embedding_model="fake",
        judge_model=STUB_JUDGE_MODEL,
        agent_model="stub-agent",
        dataset_name="questions",
    )
    with pytest.raises(AuthenticationError):
        await evaluator.run(load_full_dataset(DATASET), git_sha="test0000")
