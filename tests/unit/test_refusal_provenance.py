"""Refusal-path provenance invariant.

Guarantee that ANY refusal produced by :class:`RagAgent` carries the actual
routing decision — never ``None``, never falling back to a boot-time default
that lies about who would have served.

This is the S7 pre-tag audit fix: the mission asks for a **live** refusal
(post-S7) to be labelled honestly. Since S6 still runs the stub, we simulate
"live mode" by wiring a :class:`MultiProviderRoutingClient` in which the first
client is a stub carrying a real-looking Provider — the routing decision.
Then we verify the refusal path stamps that decision (via
``RagAgent._preferred_*``) even when synthesis is bypassed by a synthesizer
that returns an unstamped result.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from verity.agent.agent import RagAgent, _resolve_preferred, create_default_agent
from verity.agent.decomposer import LLMQuestionDecomposer
from verity.agent.scorer import LLMConfidenceScorer
from verity.agent.scripted_stub import build_stub_router
from verity.agent.synthesizer import CitedSynthesizer, SynthesisResult
from verity.api.state import AppRuntime
from verity.ingestion import (
    IngestionPipeline,
    StructureAwareChunker,
    create_default_parser,
)
from verity.ingestion.embedder import LocalEmbedder
from verity.ingestion.pipeline import discover_sources
from verity.llm.clients import StubLLMClient
from verity.llm.routing import MultiProviderRoutingClient
from verity.retrieval import InMemoryVectorStore
from verity.retrieval.hybrid import HybridRetriever
from verity.retrieval.reranker import CrossEncoderReranker
from verity.types import (
    Answer,
    Confidence,
    Provider,
    UsageStats,
)

CORPUS_DIR = Path(__file__).resolve().parents[2] / "datasets" / "corpus"


class _FakeEmbeddingBackend:
    def __init__(self, dim: int = 8) -> None:
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
        return [[0.1] * self._dim for _ in sentences]

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim


class _FakeRerankerBackend:
    def predict(self, sentences, **kw):  # type: ignore[no-untyped-def]
        return [0.5 for _ in sentences]


async def _build_retriever() -> HybridRetriever:
    embedder = LocalEmbedder(model="fake", expected_dim=8, backend=_FakeEmbeddingBackend(8))
    store = InMemoryVectorStore()
    pipeline = IngestionPipeline(
        parser=create_default_parser(),
        chunker=StructureAwareChunker(),
        embedder=embedder,
        sink=store,
    )
    result = await pipeline.ingest(discover_sources(str(CORPUS_DIR)))
    assert result.failures == []
    return HybridRetriever(
        store=store,
        embedder=embedder,
        reranker=CrossEncoderReranker(model="fake", backend=_FakeRerankerBackend()),
    )


@pytest.mark.asyncio
async def test_stub_refusal_carries_the_actual_provider() -> None:
    """Baseline: a normal stub-mode refusal (q-004 out-of-scope) already carries
    ``Provider.LOCAL`` on Answer.provider_used because the stub synthesizer stamps it.
    Prevents a regression that would let the stamp fall back to None."""
    retriever = await _build_retriever()
    agent = create_default_agent(retriever=retriever, client=build_stub_router())
    answer = await agent.answer("What is the CEO's home address?")

    assert answer.confidence.refused, "q-004 should be refused by the stub scorer"
    assert answer.provider_used is Provider.LOCAL, (
        "stub refusal must carry Provider.LOCAL — never None, never fallen back"
    )
    assert answer.model_used == "stub"


class _UnstampedSynthesizer(CitedSynthesizer):
    """Test double: emits a SynthesisResult with no provider/model stamp — the
    canonical shape of a future graceful mid-flight refusal that short-
    circuits before it can consult the router. In production this exact shape
    never occurs (the real synthesizer always stamps from the Completion), but
    we simulate it here to prove :attr:`RagAgent._preferred_*` catches the
    ``provider_used = None`` regression class."""

    async def synthesize(self, question, hits):  # type: ignore[no-untyped-def]
        return SynthesisResult(
            text="",
            citations=(),
            usage=UsageStats(),
            provider=None,
            model=None,
        )


class _AlwaysRefuseScorer(LLMConfidenceScorer):
    async def score(self, question, draft_answer, hits):  # type: ignore[no-untyped-def]
        return Confidence(
            score=0.1, refused=True, rationale="simulated mid-flight refusal for the test"
        )


@pytest.mark.asyncio
async def test_refusal_falls_back_to_preferred_routing_decision() -> None:
    """Simulate a live-mode agent whose synthesizer somehow emits an unstamped
    ``SynthesisResult``. The refusal path must fall through to the routing
    decision stamped at construction — NOT a None, NOT a silent boot-config
    default."""

    def _live_responder(_msgs):  # type: ignore[no-untyped-def]
        return '{"answer":"","citations":[]}'

    # A "live" stand-in for a real Anthropic client: same-typed provider, real
    # model id string. The test never actually calls this — it just needs the
    # routing metadata to be resolvable by ``_resolve_preferred``.
    live_client = StubLLMClient(
        _live_responder, model="claude-sonnet-4-6", provider=Provider.ANTHROPIC
    )
    router = MultiProviderRoutingClient([live_client])

    retriever = await _build_retriever()
    decomposer = LLMQuestionDecomposer(router)
    synthesizer = _UnstampedSynthesizer(router)
    scorer = _AlwaysRefuseScorer(router)

    provider, model = _resolve_preferred(router)
    agent = RagAgent(
        retriever=retriever,
        decomposer=decomposer,
        synthesizer=synthesizer,
        scorer=scorer,
        preferred_provider=provider,
        preferred_model=model,
    )

    answer = await agent.answer("What is the notice period for terminating the agreement?")

    assert answer.confidence.refused, "the simulated scorer forces refused=True"
    # The invariant: refusal carries the routing decision, not None, not boot-fallback.
    assert answer.provider_used is Provider.ANTHROPIC, (
        f"expected Anthropic (the router's first client), got {answer.provider_used!r}"
    )
    assert answer.model_used == "claude-sonnet-4-6", (
        f"expected the routing model id, got {answer.model_used!r}"
    )


def test_api_provenance_for_none_answer_labels_unknown_honestly() -> None:
    """When an Answer arrives with no provider_used (manually built, e.g. by an
    external caller or a future degraded path), ``provenance_for`` must NOT
    silently claim the boot-time default. It labels ``agent_model="unknown"``
    and preserves ``is_stub`` from the process mode as the only honest signal."""
    runtime = AppRuntime(mode="live", corpus_dir=Path("/tmp/does-not-exist"))
    unknown_answer = Answer(
        query="q",
        text="something",
        confidence=Confidence(score=0.5, refused=False, rationale=""),
        hits_used=(),
        usage=UsageStats(),
        # provider_used and model_used deliberately left as their default None.
    )
    provenance = runtime.provenance_for(unknown_answer)
    assert provenance.agent_model == "unknown"
    # is_stub reflects mode ("live") as the least-misleading fallback.
    assert provenance.is_stub is False
