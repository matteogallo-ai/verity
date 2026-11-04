"""CitedSynthesizer: quote-in-chunk enforcement + absolute-offset re-anchoring."""

from __future__ import annotations

import json
from uuid import UUID

import pytest

from verity.agent.evidence import datamark
from verity.agent.synthesizer import CitedSynthesizer, _parse_synth_output, _validate_citations
from verity.llm.clients import StubLLMClient
from verity.llm.routing import MultiProviderRoutingClient
from verity.types import (
    Chunk,
    Completion,
    DocumentId,
    Message,
    Provider,
    RetrievalHit,
    RetrieverKind,
    UsageStats,
)

DOC_ID = DocumentId("00000000-0000-0000-0000-000000000000")


def _chunk(text: str, *, cs: int, ordinal: int = 0) -> Chunk:
    return Chunk(
        id=UUID(int=ordinal + 1),
        document_id=DOC_ID,
        text=text,
        ordinal=ordinal,
        char_start=cs,
        char_end=cs + len(text),
    )


def _hit(text: str, *, cs: int, ordinal: int = 0) -> RetrievalHit:
    return RetrievalHit(
        chunk=_chunk(text, cs=cs, ordinal=ordinal),
        score=1.0 - 0.01 * ordinal,
        kind=RetrieverKind.RERANKED,
        rank=ordinal,
    )


def _router(responder) -> MultiProviderRoutingClient:  # type: ignore[no-untyped-def]
    return MultiProviderRoutingClient([StubLLMClient(responder)])


@pytest.mark.asyncio
async def test_synthesizer_validates_quote_and_reanchors_offsets() -> None:
    hits = [
        _hit("Either party may terminate on thirty (30) days notice.", cs=100, ordinal=0),
    ]
    payload = json.dumps(
        {
            "answer": "Termination requires thirty (30) days notice.",
            "citations": [{"quote": "thirty (30) days", "chunk_index": 0}],
        }
    )
    synthesizer = CitedSynthesizer(_router(lambda _: payload))
    result = await synthesizer.synthesize("q?", hits)
    assert result.citations
    citation = result.citations[0]
    quote_pos_in_chunk = hits[0].chunk.text.find("thirty (30) days")
    assert citation.char_start == hits[0].chunk.char_start + quote_pos_in_chunk
    assert citation.char_end == citation.char_start + len("thirty (30) days")
    # Chunk text is a substring of document.text (chunker invariant); the citation's quote
    # is by construction a substring of chunk.text — the assertion the acceptance
    # criteria explicitly demand.
    assert citation.quote in hits[0].chunk.text


@pytest.mark.asyncio
async def test_synthesizer_drops_hallucinated_citations() -> None:
    hits = [_hit("Foo bar baz.", cs=0)]
    payload = json.dumps(
        {
            "answer": "Something.",
            "citations": [{"quote": "not in the chunk at all", "chunk_index": 0}],
        }
    )
    synthesizer = CitedSynthesizer(_router(lambda _: payload))
    result = await synthesizer.synthesize("q?", hits)
    assert result.citations == ()


@pytest.mark.asyncio
async def test_synthesizer_drops_out_of_range_index() -> None:
    hits = [_hit("Foo bar baz.", cs=0)]
    payload = json.dumps(
        {
            "answer": "Something.",
            "citations": [{"quote": "Foo", "chunk_index": 5}],
        }
    )
    synthesizer = CitedSynthesizer(_router(lambda _: payload))
    result = await synthesizer.synthesize("q?", hits)
    assert result.citations == ()


@pytest.mark.asyncio
async def test_synthesizer_never_ships_malformed_output() -> None:
    hits = [_hit("Foo bar baz.", cs=0)]
    synthesizer = CitedSynthesizer(_router(lambda _: "definitely not JSON"))
    result = await synthesizer.synthesize("q?", hits)
    assert result.citations == ()
    assert result.text == ""


class _RecordingClient:
    """Routing client that records how the synthesizer called it. Its defaults differ
    from what the synthesizer must send, so an argument left to the default shows."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[tuple[list[Message], int]] = []

    async def complete(self, messages: list[Message], *, max_tokens: int = 1024) -> Completion:
        self.calls.append((messages, max_tokens))
        return Completion(
            text=self.text,
            provider=Provider.OPENAI,
            model="m-1",
            usage=UsageStats(input_tokens=3, output_tokens=2, llm_calls=1),
        )


@pytest.mark.asyncio
async def test_synthesizer_calls_the_model_with_its_budget_and_without_scores() -> None:
    """Drafts get a 512-token budget, and the evidence carries no retrieval scores:
    those are the scorer's input, not the drafter's. No sampling parameter is sent
    (API version 2026-10-21 rejects one)."""
    client = _RecordingClient('{"answer": "a", "citations": []}')
    synthesizer = CitedSynthesizer(client)
    assert synthesizer.last_usage is None
    result = await synthesizer.synthesize("q?", [_hit("Foo bar baz.", cs=0)])
    ((messages, max_tokens),) = client.calls
    assert max_tokens == 512
    rendered = "\n".join(m.content for m in messages)
    assert datamark("Foo bar baz.") in rendered
    assert "score=" not in rendered
    expected_usage = UsageStats(input_tokens=3, output_tokens=2, llm_calls=1)
    assert synthesizer.last_usage == result.usage == expected_usage
    assert (result.provider, result.model) == (Provider.OPENAI, "m-1")


def test_a_draft_without_an_answer_field_is_empty() -> None:
    assert _parse_synth_output('{"citations": []}') == ("", [])


def test_a_malformed_citation_entry_does_not_hide_the_ones_after_it() -> None:
    citations = ["junk", {"quote": "  ", "chunk_index": 0}, {"quote": "x", "chunk_index": 0}]
    text = json.dumps({"answer": "a", "citations": citations})
    assert _parse_synth_output(text) == ("a", [("x", 0)])


def test_a_rejected_citation_does_not_hide_the_ones_after_it() -> None:
    hits = [_hit("alpha beta", cs=10)]
    kept = _validate_citations([("alpha", 3), ("gamma", 0), ("beta", 0)], hits)
    assert [c.quote for c in kept] == ["beta"]


def test_a_repeated_quote_is_anchored_at_its_first_occurrence() -> None:
    hits = [_hit("ab ab", cs=10)]
    (citation,) = _validate_citations([("ab", 0)], hits)
    assert (citation.char_start, citation.char_end) == (10, 12)
