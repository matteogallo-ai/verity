"""End-to-end agent with stubbed LLM + in-memory retriever.

Proves the two feature-defining behaviours of S3:

- an answerable question yields a text + ≥1 citation whose ``quote`` is a substring
  of the referenced chunk;
- a below-threshold outcome collapses to an honest refusal, no fabricated text,
  ``refused = True`` on the returned :class:`Answer``.

Uses the real corpus + real chunker + a lightweight fake embedding backend so the
in-memory store has enough signal to surface the right chunks — the LLM is stubbed
end-to-end via the same scripted responder the CLI ``--stub`` flag uses.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

import pytest

from verity.agent.agent import create_default_agent
from verity.agent.scripted_stub import build_stub_router
from verity.ingestion.chunker import StructureAwareChunker
from verity.ingestion.embedder import LocalEmbedder
from verity.ingestion.parsers import create_default_parser
from verity.ingestion.pipeline import IngestionPipeline, discover_sources
from verity.llm.clients import StubLLMClient
from verity.llm.routing import MultiProviderRoutingClient
from verity.retrieval import CrossEncoderReranker, HybridRetriever, InMemoryVectorStore
from verity.types import Completion, Message, Provider, UsageStats

CORPUS = Path(__file__).resolve().parents[2] / "datasets" / "corpus"


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
    parser = create_default_parser()
    chunker = StructureAwareChunker()
    pipeline = IngestionPipeline(parser=parser, chunker=chunker, embedder=embedder, sink=store)
    result = await pipeline.ingest(discover_sources(str(CORPUS)))
    assert result.failures == []
    return HybridRetriever(
        store=store,
        embedder=embedder,
        reranker=CrossEncoderReranker(model="fake", backend=_FakeRerankerBackend()),
    )


@pytest.mark.asyncio
async def test_agent_answers_answerable_question_with_valid_citation() -> None:
    retriever = await _seed_retriever()
    agent = create_default_agent(retriever=retriever, client=build_stub_router())
    answer = await agent.answer(
        "What is the notice period for terminating the master services agreement?"
    )

    assert not answer.confidence.refused, answer.confidence.rationale
    assert answer.text
    assert answer.citations, "an answerable question must produce ≥1 citation"
    for c in answer.citations:
        # The load-bearing invariant: the citation's quote is a substring of the
        # referenced chunk, verifiable against hits_used.
        chunk = next((h.chunk for h in answer.hits_used if h.chunk.id == c.chunk_id), None)
        assert chunk is not None, "citation references a chunk not in hits_used"
        assert c.quote in chunk.text


@pytest.mark.asyncio
async def test_agent_refuses_out_of_scope_question() -> None:
    retriever = await _seed_retriever()
    agent = create_default_agent(retriever=retriever, client=build_stub_router())
    answer = await agent.answer("What is the CEO's home address?")

    assert answer.confidence.refused
    assert answer.citations == ()
    assert "don't know" in answer.text.lower()
    assert answer.confidence.rationale


class _CountingStubClient(StubLLMClient):
    """Stub client that assigns a fixed non-zero usage per role and tracks call count.

    The scripted responder wired into ``build_stub_router`` uses the system message
    to route (decompose / synthesize / confidence). We reuse the same classification
    to attach role-specific usage figures so the aggregation test can check the sum
    matches an exact expected total — no rounding, no approximation.
    """

    _USAGE_BY_KIND: ClassVar[dict[str, UsageStats]] = {
        "decompose": UsageStats(
            input_tokens=100,
            output_tokens=20,
            llm_calls=1,
            cost_usd=0.010,
            latency_ms=11.0,
        ),
        "synthesize": UsageStats(
            input_tokens=300,
            output_tokens=80,
            llm_calls=1,
            cost_usd=0.030,
            latency_ms=42.0,
        ),
        "confidence": UsageStats(
            input_tokens=250,
            output_tokens=40,
            llm_calls=1,
            cost_usd=0.020,
            latency_ms=25.0,
        ),
    }

    def __init__(self) -> None:
        from verity.agent.scripted_stub import scripted_responder

        super().__init__(scripted_responder(), model="counter", provider=Provider.LOCAL)

    async def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Completion:
        base = await super().complete(messages, max_tokens=max_tokens, temperature=temperature)
        system = (messages[0].content if messages else "").lower()
        if "decompose" in system:
            kind = "decompose"
        elif "assess whether" in system:
            kind = "confidence"
        else:
            kind = "synthesize"
        return Completion(
            text=base.text,
            provider=base.provider,
            model=base.model,
            usage=self._USAGE_BY_KIND[kind],
        )


@pytest.mark.asyncio
async def test_agent_usage_sums_every_llm_call() -> None:
    """Regression guard: Answer.usage must aggregate decompose + synthesize + score.

    On the buggy pre-fix code this asserts llm_calls==2 and misses the scorer's
    tokens/cost. Post-fix the totals must match the exact per-role figures the
    counting stub returns for a single-sub-question flow (1 decompose + 1 synthesize
    + 1 score = 3 calls).
    """
    retriever = await _seed_retriever()
    counting = _CountingStubClient()
    router = MultiProviderRoutingClient([counting])
    agent = create_default_agent(retriever=retriever, client=router)
    answer = await agent.answer("What is the notice period for terminating the agreement?")

    assert len(counting.calls) == 3, (
        f"agent must call the LLM three times (decompose + synthesize + score); "
        f"got {len(counting.calls)}"
    )
    assert answer.usage.llm_calls == 3, (
        f"Answer.usage.llm_calls must reflect every completion, got {answer.usage.llm_calls}"
    )

    expected_input = 100 + 300 + 250
    expected_output = 20 + 80 + 40
    expected_cost = 0.010 + 0.030 + 0.020
    assert answer.usage.input_tokens == expected_input, (
        f"expected {expected_input} input tokens, got {answer.usage.input_tokens}"
    )
    assert answer.usage.output_tokens == expected_output, (
        f"expected {expected_output} output tokens, got {answer.usage.output_tokens}"
    )
    assert answer.usage.cost_usd == pytest.approx(expected_cost), (
        f"expected cost_usd={expected_cost}, got {answer.usage.cost_usd}"
    )
    # Latency remains the wall-clock (measured), so we just check it's non-negative.
    assert answer.usage.latency_ms >= 0.0
    # Keep the JSON canary — the stub returns JSON so Answer.text should be non-empty.
    assert answer.text
    _ = json  # keep the import even if unused elsewhere
