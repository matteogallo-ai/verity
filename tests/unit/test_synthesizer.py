"""CitedSynthesizer: quote-in-chunk enforcement + absolute-offset re-anchoring."""

from __future__ import annotations

import hashlib
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


# Models often type the ASCII form of a mark the document prints typographically
# (French elisions and guillemets). Only these marks are interchangeable, within a class.
RSQ = "\N{RIGHT SINGLE QUOTATION MARK}"
APOSTROPHES = ("'", RSQ, "\N{LEFT SINGLE QUOTATION MARK}", "\N{MODIFIER LETTER APOSTROPHE}")
DOUBLE_QUOTES = (
    '"',
    "\N{LEFT DOUBLE QUOTATION MARK}",
    "\N{RIGHT DOUBLE QUOTATION MARK}",
    "\N{LEFT-POINTING DOUBLE ANGLE QUOTATION MARK}",
    "\N{RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK}",
)
LEASE = f"Le locataire dispose d{RSQ}un délai de cinq jours pour payer."


def test_a_straight_apostrophe_quote_finds_a_typographic_apostrophe_in_the_chunk() -> None:
    hits = [_hit(LEASE, cs=40)]
    (citation,) = _validate_citations([("dispose d'un délai de cinq jours", 0)], hits)
    assert (citation.char_start, citation.char_end) == (40 + 13, 40 + 45)


def test_the_stored_quote_keeps_the_documents_apostrophe() -> None:
    hits = [_hit(LEASE, cs=0)]
    (citation,) = _validate_citations([("dispose d'un délai", 0)], hits)
    assert citation.quote == f"dispose d{RSQ}un délai"
    assert citation.quote == LEASE[citation.char_start : citation.char_end]


@pytest.mark.parametrize("marks", [APOSTROPHES, DOUBLE_QUOTES], ids=["apostrophe", "double"])
def test_every_mark_of_a_class_matches_every_other(marks: tuple[str, ...]) -> None:
    for in_document in marks:
        hits = [_hit(f"il a dit {in_document}oui{in_document} hier", cs=0)]
        for in_quote in marks:
            (citation,) = _validate_citations([(f"dit {in_quote}oui{in_quote}", 0)], hits)
            assert citation.quote == f"dit {in_document}oui{in_document}"


@pytest.mark.parametrize(
    "quote",
    [
        "dispose d'une délai",  # a different word stays a different word
        "dispose dun délai",  # a missing apostrophe is not an apostrophe
        'dispose d"un délai',  # the two classes do not mix
        "dispose d`un délai",  # unlisted look-alikes are not folded
        "dispose d\N{ACUTE ACCENT}un délai",
    ],
)
def test_marks_outside_the_class_or_different_words_are_still_rejected(quote: str) -> None:
    hits = [_hit(LEASE, cs=0)]
    assert _validate_citations([(quote, 0)], hits) == ()


def test_an_apostrophe_does_not_match_a_double_quote_in_the_chunk() -> None:
    hits = [_hit("il a dit \N{LEFT DOUBLE QUOTATION MARK}oui\N{RIGHT DOUBLE QUOTATION MARK}", cs=0)]
    assert _validate_citations([("dit 'oui'", 0)], hits) == ()


# French typography puts a space, often a no-break one, inside guillemets; a model may
# type it or not, whatever the document does. Around a double quote mark the spacing
# is free; everywhere else words still need a gap and must match exactly.
NBSP, NNBSP = "\N{NO-BREAK SPACE}", "\N{NARROW NO-BREAK SPACE}"


@pytest.mark.parametrize(
    ("in_document", "in_quote"),
    [
        ("«de panier»", "« de panier »"),
        (f"«{NBSP}de panier{NBSP}»", "«de panier»"),
        (f"«{NNBSP}de panier{NNBSP}»", '"de panier"'),
        ("« de panier »", "«de panier »"),
    ],
)
def test_the_space_inside_guillemets_may_differ(in_document: str, in_quote: str) -> None:
    text = f"une indemnité dite {in_document} de Fr. 19.- par jour"
    hits = [_hit(text, cs=0)]
    (citation,) = _validate_citations([(f"dite {in_quote} de Fr. 19.-", 0)], hits)
    assert citation.quote == f"dite {in_document} de Fr. 19.-"


@pytest.mark.parametrize(
    "quote",
    [
        "dite « de paniers »",  # a different word stays a different word
        "dite « depanier »",  # words still need their gap
    ],
)
def test_free_spacing_around_guillemets_does_not_loosen_the_words(quote: str) -> None:
    hits = [_hit("une indemnité dite «de panier» par jour", cs=0)]
    assert _validate_citations([(quote, 0)], hits) == ()


# Asked to name the article, a draft lets a member find the provision in the text; off
# by default, so the default prompt is the one the published runs used.
async def _system_prompt(synthesizer: CitedSynthesizer, client: _RecordingClient) -> str:
    await synthesizer.synthesize("q?", [_hit("Art. 8.1 Foo bar baz.", cs=0)])
    ((messages, _),) = client.calls
    return messages[0].content


@pytest.mark.asyncio
async def test_by_default_the_draft_is_not_asked_to_name_the_article() -> None:
    client = _RecordingClient('{"answer": "a", "citations": []}')
    assert "provision" not in await _system_prompt(CitedSynthesizer(client), client)


@pytest.mark.asyncio
async def test_the_setting_asks_the_draft_to_name_the_article(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from verity.config import get_settings

    monkeypatch.setenv("VERITY_NAME_ARTICLES", "true")
    get_settings.cache_clear()
    client = _RecordingClient('{"answer": "a", "citations": []}')
    system = await _system_prompt(CitedSynthesizer(client), client)
    assert "name the article" in system
    assert system.startswith(
        await _system_prompt(
            CitedSynthesizer(off := _RecordingClient("{}"), name_articles=False), off
        )
    )


@pytest.mark.asyncio
async def test_the_default_system_prompt_is_the_one_the_published_runs_used() -> None:
    """Pinned on the template of 9165978, before the article note existed: the file's
    hash changed with the placeholder, the rendered prompt must not."""
    client = _RecordingClient('{"answer": "a", "citations": []}')
    system = await _system_prompt(CitedSynthesizer(client, name_articles=False), client)
    assert hashlib.sha256(system.encode()).hexdigest() == (
        "a80cccad4fbffd791561ca3347d5c3ffa8acda2a8a67c3f094f1b14b921af2c9"
    )


@pytest.mark.asyncio
async def test_the_article_note_is_a_paragraph_of_its_own() -> None:
    client = _RecordingClient('{"answer": "a", "citations": []}')
    system = await _system_prompt(CitedSynthesizer(client, name_articles=True), client)
    assert "\n\nWhen the documents number" in system


# A PDF prints an ordinal's suffix as a superscript, which text extraction separates
# from its number ("au 1 er janvier"); a model quotes it as written ("1er"). The gap
# between a number and its ordinal suffix is free; everywhere else it is not.
@pytest.mark.parametrize(
    ("in_document", "in_quote"),
    [
        ("au 1 er janvier", "au 1er janvier"),
        ("au 1er janvier", "au 1 er janvier"),
        ("de la 2 e année", "de la 2e année"),
        ("dès la 3 ème année", "dès la 3ème année"),
        ("la 1 re année,", "la 1re année,"),
        ("la 1 ère année", "la 1ère année"),
    ],
)
def test_the_gap_between_a_number_and_its_ordinal_suffix_may_differ(
    in_document: str, in_quote: str
) -> None:
    text = f"Le passage est effectif {in_document} qui suit."
    hits = [_hit(text, cs=0)]
    (citation,) = _validate_citations([(f"effectif {in_quote}", 0)], hits)
    assert citation.quote == f"effectif {in_document}"


@pytest.mark.parametrize(
    "quote",
    [
        "au 2er janvier",  # a different number stays a different number
        "au 1 ex janvier",  # only an ordinal suffix joins its number
        "au1er janvier",  # the word before the number still needs its gap
    ],
)
def test_a_free_ordinal_gap_does_not_loosen_the_words(quote: str) -> None:
    hits = [_hit("effectif au 1 er janvier", cs=0)]
    assert _validate_citations([(quote, 0)], hits) == ()


@pytest.mark.parametrize(
    ("in_document", "quote"),
    [("envoyer 2 e-mail", "envoyer 2e-mail"), ("lettre 2 e)", "lettre 2e)"), ("1 e5", "1e5")],
)
def test_only_a_whole_ordinal_suffix_joins_its_number(in_document: str, quote: str) -> None:
    assert _validate_citations([(quote, 0)], [_hit(in_document, cs=0)]) == ()


@pytest.mark.asyncio
async def test_every_quote_the_draft_proposed_is_kept_with_whether_it_was_found() -> None:
    """A refused answer drops its citations; the proposed quotes say why one was rejected."""
    hits = [_hit("Either party may terminate on thirty (30) days notice.", cs=0)]
    payload = json.dumps(
        {
            "answer": "Thirty days.",
            "citations": [
                {"quote": "thirty (30) days", "chunk_index": 0},
                {"quote": "sixty days", "chunk_index": 0},
                {"quote": "thirty", "chunk_index": 4},
            ],
        }
    )
    result = await CitedSynthesizer(_router(lambda _: payload)).synthesize("q?", hits)
    assert [(p.quote, p.chunk_index, p.verified) for p in result.proposed] == [
        ("thirty (30) days", 0, True),
        ("sixty days", 0, False),
        ("thirty", 4, False),
    ]
