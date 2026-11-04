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

import re
from dataclasses import dataclass

from verity.agent.evidence import DATAMARK, DATAMARK_NOTE, neutralise, render_evidence
from verity.llm.base import RoutingClient
from verity.llm.json_output import parse_json_object
from verity.llm.prompt import load_prompt
from verity.types import Citation, Provider, RetrievalHit, UsageStats


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


class CitedSynthesizer:
    """Produces a draft answer + validated ``Citation``s from a ranked evidence set."""

    def __init__(
        self,
        client: RoutingClient,
        *,
        prompt_name: str = "synthesize",
        max_tokens: int = 512,
        datamarked: bool = True,
    ) -> None:
        self._client = client
        # Off only to measure what the mark changes (scripts/run_injection_eval.py).
        self._datamarked = datamarked
        self._prompt = load_prompt(prompt_name)
        self._max_tokens = max_tokens
        self.last_usage: UsageStats | None = None

    async def synthesize(self, question: str, hits: list[RetrievalHit]) -> SynthesisResult:
        evidence = render_evidence(hits, with_scores=False, datamarked=self._datamarked)
        messages = self._prompt.render(
            question=neutralise(question),
            evidence=evidence,
            datamark_note=DATAMARK_NOTE if self._datamarked else "",
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
        )


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


def _locate(quote: str, text: str) -> tuple[int, int] | None:
    """The first span of ``text`` holding ``quote``'s words in order, separated by any
    whitespace; ``None`` if there is none."""
    words = [w for w in _WORD_GAP_RE.split(quote) if w]
    if not words:
        return None
    # A literal mark in a document reaches the model as "^" (see evidence.datamark).
    pattern = r"\s+".join(re.escape(w).replace(r"\^", f"[\\^{DATAMARK}]") for w in words)
    match = re.search(pattern, text)
    return None if match is None else (match.start(), match.end())


__all__ = ["CitedSynthesizer", "SynthesisResult"]
