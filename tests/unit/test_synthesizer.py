"""CitedSynthesizer: quote-in-chunk enforcement + absolute-offset re-anchoring."""

from __future__ import annotations

import json
from uuid import UUID

import pytest

from verity.agent.synthesizer import CitedSynthesizer
from verity.llm.clients import StubLLMClient
from verity.llm.routing import MultiProviderRoutingClient
from verity.types import Chunk, DocumentId, RetrievalHit, RetrieverKind

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
    # Chunk text is a substring of document.text (S1 invariant); the citation's quote
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
async def test_synthesizer_falls_open_on_malformed_output() -> None:
    hits = [_hit("Foo bar baz.", cs=0)]
    synthesizer = CitedSynthesizer(_router(lambda _: "definitely not JSON"))
    result = await synthesizer.synthesize("q?", hits)
    assert result.citations == ()
    assert "not JSON" in result.text
