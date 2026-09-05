"""Smoke tests that the stage contracts import cleanly and are structurally checkable.

These do not test behaviour (there is none yet at S0) — they lock the *shape* of the
interfaces so later stages implement against a stable surface.
"""

from __future__ import annotations

from verity.agent.base import Agent, ConfidenceScorer, QuestionDecomposer
from verity.eval.base import Evaluator, FaithfulnessJudge, RetrievalMetric, RunStore
from verity.ingestion.base import Chunker, Embedder, Parser
from verity.llm.base import (
    AuthenticationError,
    LLMClient,
    LLMError,
    ProviderUnavailableError,
    RoutingClient,
)
from verity.observability.base import StageSpan, Tracer
from verity.retrieval.base import Fusion, Reranker, Retriever, VectorStore


def test_all_stage_protocols_import() -> None:
    protocols = [
        Parser,
        Chunker,
        Embedder,
        VectorStore,
        Retriever,
        Fusion,
        Reranker,
        LLMClient,
        RoutingClient,
        QuestionDecomposer,
        ConfidenceScorer,
        Agent,
        RetrievalMetric,
        FaithfulnessJudge,
        Evaluator,
        RunStore,
        StageSpan,
        Tracer,
    ]
    assert all(p is not None for p in protocols)


def test_llm_error_hierarchy() -> None:
    # The routing contract depends on AuthenticationError being distinguishable from
    # a transient failure — assert the hierarchy the RoutingClient will branch on.
    assert issubclass(AuthenticationError, LLMError)
    assert issubclass(ProviderUnavailableError, LLMError)
    assert not issubclass(AuthenticationError, ProviderUnavailableError)


def test_runtime_checkable_structural_conformance() -> None:
    # runtime_checkable protocols accept a structurally-conforming object.
    class FakeEmbedder:
        @property
        def model(self) -> str:
            return "fake"

        @property
        def dim(self) -> int:
            return 3

        async def embed(self, texts: list[str]) -> list[tuple[float, ...]]:
            return [(0.0, 0.0, 0.0) for _ in texts]

        async def embed_chunks(self, chunks: list[object]) -> list[object]:
            return []

    assert isinstance(FakeEmbedder(), Embedder)
