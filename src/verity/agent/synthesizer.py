"""Cited-answer synthesis.

The LLM is asked (via ``synthesize.prompt``) to return JSON of the form
``{"answer": "...", "citations": [{"quote": "...", "chunk_index": N}]}``.

We do **not** trust the LLM to invent citations. Every citation is:

1. mapped to the ``N``-th chunk in the evidence list we handed to the model (indexes
   into the ``hits`` argument, not into any global chunk table);
2. rejected if its ``quote`` is not a verbatim substring of the referenced
   ``chunk.text`` (the invariant that citations point at real passages);
3. re-anchored so ``char_start`` / ``char_end`` are offsets into
   ``document.text`` (chunk.char_start + quote position), which is what the UI
   highlights against.

Drafting is intentionally decoupled from confidence scoring — this module returns
the draft text + validated citations, and ``verity.agent.scorer`` decides whether
the draft ships or is replaced by a refusal.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass

from verity.agent.evidence import DATAMARK, DATAMARK_NOTE, neutralise, render_evidence
from verity.config import get_settings
from verity.llm.base import RoutingClient
from verity.llm.json_output import parse_json_object
from verity.llm.prompt import load_prompt
from verity.types import Citation, Provider, RetrievalHit, UsageStats

DRAFT_MAX_TOKENS = 512

# Appended to the system message with VERITY_NAME_ARTICLES. The number comes from the
# quoted text: a chunk's section attribute can name a neighbouring article.
ARTICLE_NOTE = (
    "\n\nWhen the documents number their provisions (articles, sections, clauses), "
    "name the article each claim comes from in the answer, as the cited text numbers it "
    '(for example "art. 8.1").'
)


# Appended to the system message with VERITY_COMPLETE_ANSWERS. The uncovered part goes
# in its own field: the agent appends it after the support check, which a sentence
# citing nothing would otherwise fail.
COMPLETE_NOTE = (
    "\n\nAnswer every part of the question. When the documents answer only some parts, "
    'answer those in "answer", and put one short sentence naming each part they do not '
    'cover, in the language of the question, in a "not_covered" field; leave it out when '
    "nothing is missing. Copy every amount, duration, date and percentage exactly as the "
    "cited text writes it: do not convert, round or reword it."
)


@dataclass(frozen=True)
class SynthesisResult:
    """Output of the synthesizer — draft text, validated citations, usage, and the
    concrete LLM provider/model that produced ``text``.

    The synthesizer's ``Completion`` is what carries the actual provider a
    :class:`RoutingClient` served the request with (including any fallback the
    router performed). We surface it here so :class:`RagAgent` can stamp
    :attr:`Answer.provider_used` / :attr:`Answer.model_used` without needing
    to re-query the router. This is the load-bearing signal for per-request
    provenance in the UI.

    ``provider`` and ``model`` may be ``None`` when a synthesizer returns
    without consulting the router (the test double in
    ``tests/unit/test_refusal_provenance.py`` does). In that case the RagAgent's
    ``preferred_*`` fallback stamps the routing decision — the invariant
    covered by ``tests/unit/test_refusal_provenance.py``.
    """

    text: str
    citations: tuple[Citation, ...]
    usage: UsageStats
    provider: Provider | None
    model: str | None
    # What the documents do not cover, as the draft names it (VERITY_COMPLETE_ANSWERS).
    not_covered: str | None = None


class CitedSynthesizer:
    """Produces a draft answer + validated ``Citation``s from a ranked evidence set."""

    def __init__(
        self,
        client: RoutingClient,
        *,
        prompt_name: str = "synthesize",
        max_tokens: int = DRAFT_MAX_TOKENS,
        datamarked: bool = True,
        name_articles: bool | None = None,
        complete_answers: bool | None = None,
    ) -> None:
        self._client = client
        self._name_articles = (
            get_settings().name_articles if name_articles is None else name_articles
        )
        self._complete_answers = (
            get_settings().complete_answers if complete_answers is None else complete_answers
        )
        # Off only to measure what the mark changes (scripts/run_injection_eval.py).
        self._datamarked = datamarked
        self._prompt = load_prompt(prompt_name)
        self._max_tokens = max_tokens
        self.last_usage: UsageStats | None = None

    @property
    def max_tokens(self) -> int:
        """The draft's output cap: a draft cut off by it is unparseable and refused."""
        return self._max_tokens

    async def synthesize(self, question: str, hits: list[RetrievalHit]) -> SynthesisResult:
        evidence = render_evidence(hits, with_scores=False, datamarked=self._datamarked)
        messages = self._prompt.render(
            question=neutralise(question),
            evidence=evidence,
            datamark_note=DATAMARK_NOTE if self._datamarked else "",
            article_note=ARTICLE_NOTE if self._name_articles else "",
            answer_note=COMPLETE_NOTE if self._complete_answers else "",
        )
        completion = await self._client.complete(messages, max_tokens=self._max_tokens)
        self.last_usage = completion.usage
        text, raw_citations = _parse_synth_output(completion.text)
        citations = _validate_citations(raw_citations, hits)
        return SynthesisResult(
            text=text,
            citations=citations,
            usage=completion.usage,
            provider=completion.provider,
            model=completion.model,
            not_covered=_parse_not_covered(completion.text) if self._complete_answers else None,
        )


def _parse_not_covered(text: str) -> str | None:
    parsed = parse_json_object(text)
    value = parsed.get("not_covered") if parsed else None
    if not isinstance(value, str):
        return None
    return value.replace(DATAMARK, " ").strip() or None


def _parse_synth_output(text: str) -> tuple[str, list[tuple[str, int]]]:
    """Extract ``(answer, [(quote, chunk_index), ...])`` from the LLM's JSON output.

    Robust to prose wrapping, markdown code fences, and single-element array
    wrappers (all real-model drifts observed with Sonnet-class outputs). On
    total parse failure return ``("", [])``: the raw model text is never shipped,
    and the agent refuses the empty, uncited draft with reason ``no_citation``.
    """
    parsed = parse_json_object(text)
    if parsed is None:
        return "", []
    # A model that echoes marked evidence must not show the mark to the user.
    answer = str(parsed.get("answer", "")).replace(DATAMARK, " ").strip()
    citations_raw = parsed.get("citations", []) or []
    citations: list[tuple[str, int]] = []
    if isinstance(citations_raw, list):
        for entry in citations_raw:
            if not isinstance(entry, dict):
                continue
            quote = entry.get("quote")
            idx = entry.get("chunk_index")
            if isinstance(quote, str) and isinstance(idx, int):
                stripped = quote.strip()
                if stripped:
                    citations.append((stripped, idx))
    return answer, citations


def _validate_citations(
    raw_citations: list[tuple[str, int]],
    hits: list[RetrievalHit],
) -> tuple[Citation, ...]:
    """Keep only citations whose quote is in the referenced chunk, word for word.

    The model reads datamarked evidence, so a quote may join its words with the mark,
    with spaces, or keep a line break the evidence no longer shows: words must match
    exactly and in order, whitespace and marks between them may differ. The citation
    stores the chunk's own text for that span, so ``quote`` is always a verbatim
    substring of the document.

    ``char_start``/``char_end`` are re-anchored to ``document.text`` offsets — the
    chunker guarantees ``document.text[chunk.char_start:chunk.char_end] == chunk.text``,
    so the absolute span is ``chunk.char_start + position_of_quote_in_chunk``.
    """
    out: list[Citation] = []
    for quote, idx in raw_citations:
        if idx < 0 or idx >= len(hits):
            continue
        chunk = hits[idx].chunk
        span = _locate(quote, chunk.text)
        if span is None:
            # LLM invented or paraphrased — reject rather than let a bogus offset ship.
            continue
        start, end = span
        abs_start = chunk.char_start + start
        abs_end = chunk.char_start + end
        out.append(
            Citation(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                quote=chunk.text[start:end],
                char_start=abs_start,
                char_end=abs_end,
            )
        )
    return tuple(out)


_WORD_GAP_RE = re.compile(rf"[\s{DATAMARK}]+")

# A model often types the ASCII form of a mark the document prints typographically
# (French elisions and guillemets), which made a correct quote unfindable. Each mark
# matches only the marks of its own class; every other character must match exactly.
_APOSTROPHES = (
    "'\N{RIGHT SINGLE QUOTATION MARK}\N{LEFT SINGLE QUOTATION MARK}\N{MODIFIER LETTER APOSTROPHE}"
)
_DOUBLE_QUOTES = (
    '"'
    "\N{LEFT DOUBLE QUOTATION MARK}"
    "\N{RIGHT DOUBLE QUOTATION MARK}"
    "\N{LEFT-POINTING DOUBLE ANGLE QUOTATION MARK}"
    "\N{RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK}"
)
_EQUIVALENT_MARKS = {
    mark: f"[{re.escape(marks)}]" for marks in (_APOSTROPHES, _DOUBLE_QUOTES) for mark in marks
}
# A literal mark in a document reaches the model as "^" (see evidence.datamark).
_EQUIVALENT_MARKS["^"] = f"[\\^{DATAMARK}]"


def _word_pattern(word: str) -> str:
    return "".join(_EQUIVALENT_MARKS.get(char) or re.escape(char) for char in word)


# French typography spaces the inside of guillemets, often with a no-break space, and a
# model may type that space or not whatever the document does: next to a double quote
# mark the gap is optional. Between two words it is still required.
_DOUBLE_QUOTE_SPLIT_RE = re.compile(f"([{re.escape(_DOUBLE_QUOTES)}])")


def _locate(quote: str, text: str) -> tuple[int, int] | None:
    """The first span of ``text`` holding ``quote``'s words in order, separated by any
    whitespace, where an apostrophe or a double quote matches any mark of its class
    and the space next to a double quote mark is optional; ``None`` if there is none."""
    tokens = [
        piece
        for word in _WORD_GAP_RE.split(quote)
        for piece in _DOUBLE_QUOTE_SPLIT_RE.split(word)
        if piece
    ]
    if not tokens:
        return None
    pattern = _word_pattern(tokens[0])
    for before, token in itertools.pairwise(tokens):
        gap = r"\s*" if before in _DOUBLE_QUOTES or token in _DOUBLE_QUOTES else r"\s+"
        pattern += gap + _word_pattern(token)
    match = re.search(pattern, text)
    return None if match is None else (match.start(), match.end())


__all__ = ["CitedSynthesizer", "SynthesisResult"]
