"""An infrastructure failure is a typed error, never a refusal; an uncited draft is
always a refusal (reason ``no_citation``), never an answer."""

from __future__ import annotations

import json
from typing import Any

import psycopg
import pytest

from tests.unit.test_agent import _CountingStubClient, _seed_retriever
from verity.agent.agent import create_default_agent
from verity.agent.scorer import LLMConfidenceScorer
from verity.agent.scripted_stub import scripted_responder
from verity.llm.base import AuthenticationError, LLMError, ProviderUnavailableError
from verity.llm.clients import StubLLMClient
from verity.llm.routing import MultiProviderRoutingClient
from verity.retrieval import HybridRetriever, InMemoryVectorStore
from verity.retrieval.base import RetrievalUnavailableError
from verity.types import (
    NO_CITATION_RATIONALE,
    REFUSAL_TEXT,
    Answer,
    AnswerError,
    Completion,
    Confidence,
    ErrorKind,
    Message,
    RefusalReason,
)

_ANSWERABLE = "What is the notice period for terminating the master services agreement?"


# --- Answer invariants ----------------------------------------------------------------


def test_uncited_unrefused_answer_becomes_a_no_citation_refusal() -> None:
    answer = Answer(
        query="q",
        text="A fluent answer with nothing behind it.",
        confidence=Confidence(score=0.97, refused=False, rationale="looks fine"),
    )
    assert answer.confidence is not None
    assert answer.confidence.refused
    assert answer.confidence.reason is RefusalReason.NO_CITATION
    assert answer.confidence.rationale == NO_CITATION_RATIONALE
    assert answer.text == REFUSAL_TEXT


def test_a_naive_baseline_answer_ships_uncited() -> None:
    answer = Answer(
        query="q",
        text="A fluent answer with nothing behind it.",
        confidence=Confidence(score=1.0, refused=False, rationale="checks disabled"),
        checks_disabled=True,
    )
    assert answer.confidence is not None and not answer.confidence.refused
    assert answer.text == "A fluent answer with nothing behind it."


def test_answer_needs_exactly_one_of_confidence_or_error() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        Answer(query="q", text="")
    with pytest.raises(ValueError, match="exactly one"):
        Answer(
            query="q",
            text="",
            confidence=Confidence(score=0.0, refused=True, rationale="r"),
            error=AnswerError(kind=ErrorKind.PROVIDER_UNAVAILABLE, detail="d"),
        )


# --- Agent: no_citation gate ----------------------------------------------------------


def _router_with_synthesis(synthesis_text: str) -> tuple[MultiProviderRoutingClient, list[str]]:
    """Scripted stub for every role except synthesis, which returns ``synthesis_text``.
    Records which roles were called."""
    scripted = scripted_responder()
    roles: list[str] = []

    def _respond(messages: list[Message]) -> str:
        system = messages[0].content.lower()
        if "draft concise" in system:
            roles.append("synthesize")
            return synthesis_text
        roles.append("confidence" if "assess whether" in system else "decompose")
        return scripted(messages)

    return MultiProviderRoutingClient([StubLLMClient(_respond)]), roles


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "synthesis_text",
    [
        "Sure! The notice period is thirty days.",  # not JSON: never shipped
        json.dumps({"answer": "Thirty days.", "citations": []}),  # JSON, no citation
        json.dumps(  # citation whose quote is not in the chunk: dropped
            {"answer": "Thirty days.", "citations": [{"quote": "ninety", "chunk_index": 0}]}
        ),
    ],
    ids=["unparseable", "no-citations", "invented-quote"],
)
async def test_agent_refuses_an_uncited_draft_without_asking_the_scorer(
    synthesis_text: str,
) -> None:
    router, roles = _router_with_synthesis(synthesis_text)
    agent = create_default_agent(retriever=await _seed_retriever(), client=router)

    answer = await agent.answer(_ANSWERABLE)

    assert answer.error is None
    assert answer.confidence is not None
    assert answer.confidence.refused
    assert answer.confidence.reason is RefusalReason.NO_CITATION
    assert answer.text == REFUSAL_TEXT
    assert "confidence" not in roles, "an uncited draft must not reach the scorer"


@pytest.mark.asyncio
async def test_scorer_reports_why_it_refused() -> None:
    def _scorer(payload: object) -> LLMConfidenceScorer:
        text = payload if isinstance(payload, str) else json.dumps(payload)
        return LLMConfidenceScorer(
            MultiProviderRoutingClient([StubLLMClient(lambda _: text)]), threshold=0.55
        )

    below = await _scorer({"score": 0.2, "refused": False, "rationale": "thin"}).score("q", "a", [])
    model = await _scorer({"score": 0.9, "refused": True, "rationale": "off"}).score("q", "a", [])
    broken = await _scorer("no json").score("q", "a", [])
    fine = await _scorer({"score": 0.9, "refused": False, "rationale": "ok"}).score("q", "a", [])
    assert below.reason is RefusalReason.BELOW_THRESHOLD
    assert model.reason is RefusalReason.MODEL_REFUSED
    assert broken.reason is RefusalReason.SCORER_UNPARSEABLE
    assert fine.reason is None and not fine.refused


# --- Agent: infrastructure failures ---------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exc", "kind"),
    [
        (ProviderUnavailableError("all providers down"), ErrorKind.PROVIDER_UNAVAILABLE),
        (AuthenticationError("bad key"), ErrorKind.PROVIDER_AUTH),
        (LLMError("400 bad request"), ErrorKind.PROVIDER_ERROR),
    ],
)
async def test_provider_failure_is_an_error_not_a_refusal(exc: Exception, kind: ErrorKind) -> None:
    def _raise(_messages: list[Message]) -> str:
        raise exc

    router = MultiProviderRoutingClient([StubLLMClient(_raise)])
    agent = create_default_agent(retriever=await _seed_retriever(), client=router)

    answer = await agent.answer(_ANSWERABLE)

    assert answer.confidence is None
    assert answer.error is not None
    assert answer.error.kind is kind
    assert answer.text == ""
    assert answer.citations == ()


class _DownStore(InMemoryVectorStore):
    async def dense_search(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise ConnectionError("connection refused (store down)")


@pytest.mark.asyncio
async def test_a_failed_retrieval_stage_raises_instead_of_returning_nothing() -> None:
    seeded = await _seed_retriever()
    retriever = HybridRetriever(store=_DownStore(), embedder=seeded._embedder, reranker=None)
    with pytest.raises(RetrievalUnavailableError, match="store down"):
        await retriever.retrieve("notice period", k=3)


class _BuggyStore(InMemoryVectorStore):
    async def dense_search(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise TypeError("unsupported operand type(s)")


class _QueryFailsStore(InMemoryVectorStore):
    async def dense_search(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise psycopg.errors.UndefinedTable('relation "chunks" does not exist')


class _UnmigratedStore(InMemoryVectorStore):
    async def dense_search(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        from verity.retrieval.pgvector_store import SchemaNotMigratedError

        raise SchemaNotMigratedError("chunks table missing: run `verity db upgrade`")


@pytest.mark.asyncio
async def test_an_unmigrated_store_is_retrieval_unavailable_not_a_defect() -> None:
    seeded = await _seed_retriever()
    retriever = HybridRetriever(store=_UnmigratedStore(), embedder=seeded._embedder, reranker=None)
    with pytest.raises(RetrievalUnavailableError, match="verity db upgrade"):
        await retriever.retrieve("notice period", k=3)


@pytest.mark.asyncio
async def test_a_bug_in_a_retrieval_stage_is_not_reported_as_the_store_being_down() -> None:
    seeded = await _seed_retriever()
    retriever = HybridRetriever(store=_BuggyStore(), embedder=seeded._embedder, reranker=None)
    with pytest.raises(TypeError, match="unsupported operand"):
        await retriever.retrieve("notice period", k=3)


@pytest.mark.asyncio
async def test_a_failed_store_query_is_still_retrieval_unavailable() -> None:
    seeded = await _seed_retriever()
    retriever = HybridRetriever(store=_QueryFailsStore(), embedder=seeded._embedder, reranker=None)
    with pytest.raises(RetrievalUnavailableError, match="chunks"):
        await retriever.retrieve("notice period", k=3)


@pytest.mark.asyncio
async def test_store_down_is_an_error_not_a_refusal() -> None:
    seeded = await _seed_retriever()
    retriever = HybridRetriever(store=_DownStore(), embedder=seeded._embedder, reranker=None)
    router = MultiProviderRoutingClient([StubLLMClient(scripted_responder())])
    agent = create_default_agent(retriever=retriever, client=router)

    answer = await agent.answer(_ANSWERABLE)

    assert answer.confidence is None
    assert answer.error is not None
    assert answer.error.kind is ErrorKind.RETRIEVAL_UNAVAILABLE
    assert "store down" in answer.error.detail


class _ScorerDownClient(_CountingStubClient):
    """Decompose and synthesize succeed and are billed; the scorer's provider is down."""

    async def complete(self, messages: list[Message], **kwargs: Any) -> Completion:
        if "assess whether" in messages[0].content.lower():
            raise ProviderUnavailableError("scorer provider down")
        return await super().complete(messages, **kwargs)


@pytest.mark.asyncio
async def test_an_errored_answer_still_reports_the_tokens_it_spent() -> None:
    """A spending cap counts errored answers: their usage must not read as zero."""
    agent = create_default_agent(
        retriever=await _seed_retriever(),
        client=MultiProviderRoutingClient([_ScorerDownClient()]),
    )

    answer = await agent.answer(_ANSWERABLE)

    assert answer.error is not None
    assert answer.error.kind is ErrorKind.PROVIDER_UNAVAILABLE
    assert answer.usage.llm_calls == 2
    assert answer.usage.input_tokens == 100 + 300
    assert answer.usage.output_tokens == 20 + 80
    assert answer.usage.cost_usd == pytest.approx(0.010 + 0.030)


@pytest.mark.asyncio
async def test_a_bug_after_a_billed_call_is_an_internal_error_that_keeps_its_spend() -> None:
    """The decompose call is paid before retrieval runs: a bug there must not lose it."""
    seeded = await _seed_retriever()
    retriever = HybridRetriever(store=_BuggyStore(), embedder=seeded._embedder, reranker=None)
    agent = create_default_agent(
        retriever=retriever, client=MultiProviderRoutingClient([_CountingStubClient()])
    )

    answer = await agent.answer(_ANSWERABLE)

    assert answer.error is not None
    assert answer.error.kind is ErrorKind.INTERNAL
    assert answer.usage.llm_calls == 1
    assert answer.usage.cost_usd > 0


class _BrokenModel:
    """A local model whose weights cannot be loaded or whose inference fails."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def embed(self, texts: list[str]) -> list[tuple[float, ...]]:
        raise self._exc

    async def rerank(self, query: str, hits: list[Any], top_k: int) -> list[Any]:
        raise self._exc


@pytest.mark.parametrize("stage", ["embedder", "reranker"])
@pytest.mark.parametrize(
    "exc", [OSError("model not in the cache"), RuntimeError("CUDA out of memory")]
)
@pytest.mark.asyncio
async def test_a_local_model_failure_is_a_typed_error_that_keeps_its_spend(
    stage: str, exc: Exception
) -> None:
    seeded = await _seed_retriever()
    broken = _BrokenModel(exc)
    retriever = HybridRetriever(
        store=seeded._store,
        embedder=broken if stage == "embedder" else seeded._embedder,  # type: ignore[arg-type]
        reranker=broken if stage == "reranker" else seeded._reranker,  # type: ignore[arg-type]
    )
    agent = create_default_agent(
        retriever=retriever, client=MultiProviderRoutingClient([_CountingStubClient()])
    )

    answer = await agent.answer(_ANSWERABLE)

    assert answer.confidence is None
    assert answer.error is not None
    assert answer.error.kind is ErrorKind.MODEL_UNAVAILABLE
    assert str(exc) in answer.error.detail
    # The decompose call was paid before retrieval failed.
    assert answer.usage.llm_calls == 1 and answer.usage.cost_usd > 0


@pytest.mark.asyncio
async def test_an_embedder_that_returns_no_vector_is_a_model_error() -> None:
    seeded = await _seed_retriever()

    class _Empty:
        async def embed(self, texts: list[str]) -> list[tuple[float, ...]]:
            return []

    retriever = HybridRetriever(store=seeded._store, embedder=_Empty(), reranker=None)  # type: ignore[arg-type]
    agent = create_default_agent(
        retriever=retriever, client=MultiProviderRoutingClient([_CountingStubClient()])
    )
    answer = await agent.answer(_ANSWERABLE)
    assert answer.error is not None
    assert answer.error.kind is ErrorKind.MODEL_UNAVAILABLE
