"""Cited-answer synthesis.

The LLM is asked (via ``synthesize.prompt``) to return JSON of the form
``{"answer": "...", "citations": [{"quote": "...", "chunk_index": N}]}``.

We do **not** trust the LLM to invent citations. Every citation is:

1. mapped to the ``N``-th chunk in the evidence list we handed to the model (indexes
   into the ``hits`` argument, not into any global chunk table);
2. rejected if its ``quote`` is not a verbatim substring of the referenced
   ``chunk.text`` (the load-bearing S1 invariant that citations point at real
   passages);
3. re-anchored so ``char_start`` / ``char_end`` are offsets into
   ``document.text`` (chunk.char_start + quote position), which is what the UI
   highlights against.

Drafting is intentionally decoupled from confidence scoring — this module returns
the draft text + validated citations, and ``verity.agent.scorer`` decides whether
the draft ships or is replaced by a refusal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from verity.config import get_settings
from verity.llm.base import RoutingClient
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

    ``provider`` and ``model`` may be ``None`` when a future degraded synthesis
    path (e.g. graceful mid-flight refusal on a transient LLM error) short-
    circuits before it can consult the router. In that case the RagAgent's
    ``preferred_*`` fallback stamps the honest routing decision — the invariant
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
    ) -> None:
        self._client = client
        self._prompt = load_prompt(prompt_name)
        self._max_tokens = max_tokens
        self.last_usage: UsageStats | None = None

    async def synthesize(self, question: str, hits: list[RetrievalHit]) -> SynthesisResult:
        settings = get_settings()  # noqa: F841 — reserved for future max-hit trimming knobs
        evidence = _render_evidence(hits)
        messages = self._prompt.render(question=question, evidence=evidence)
        completion = await self._client.complete(
            messages, max_tokens=self._max_tokens, temperature=0.0
        )
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


def _render_evidence(hits: list[RetrievalHit]) -> str:
    lines: list[str] = []
    for i, hit in enumerate(hits):
        section = hit.chunk.section or "-"
        # Strip trailing whitespace so the LLM sees a clean block per chunk.
        body = hit.chunk.text.strip()
        lines.append(f"[{i}] (section: {section})\n{body}")
    return "\n\n".join(lines) if lines else "(no evidence available)"


def _parse_synth_output(text: str) -> tuple[str, list[tuple[str, int]]]:
    """Extract ``(answer, [(quote, chunk_index), ...])`` from the LLM's JSON output.

    Robust to prose wrapping, markdown code fences, and single-element array
    wrappers (all real-model drifts observed with Sonnet-class outputs). On
    total parse failure return ``(text.strip(), [])`` so the citation validator
    drops the citations and the scorer refuses — safe path.
    """
    candidate = _first_json_object(text) or text
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return text.strip(), []
    if isinstance(parsed, list):
        parsed = next((item for item in parsed if isinstance(item, dict)), None)
    if not isinstance(parsed, dict):
        return text.strip(), []
    answer = str(parsed.get("answer", "")).strip()
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


def _first_json_object(text: str) -> str:
    """Return the largest ``{...}`` slice, or the first ``[...]`` slice as a
    fallback for models that wrap the response in a single-element array."""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    array_start = text.find("[")
    array_end = text.rfind("]")
    if array_start != -1 and array_end != -1 and array_end > array_start:
        return text[array_start : array_end + 1]
    return ""


def _validate_citations(
    raw_citations: list[tuple[str, int]],
    hits: list[RetrievalHit],
) -> tuple[Citation, ...]:
    """Keep only citations whose quote is a verbatim substring of the referenced chunk.

    ``char_start``/``char_end`` are re-anchored to ``document.text`` offsets — the S1
    contract is that ``document.text[chunk.char_start:chunk.char_end] == chunk.text``,
    so the absolute span is ``chunk.char_start + position_of_quote_in_chunk``.
    """
    out: list[Citation] = []
    for quote, idx in raw_citations:
        if idx < 0 or idx >= len(hits):
            continue
        chunk = hits[idx].chunk
        pos = chunk.text.find(quote)
        if pos < 0:
            # LLM invented or paraphrased — reject rather than let a bogus offset ship.
            continue
        abs_start = chunk.char_start + pos
        abs_end = abs_start + len(quote)
        out.append(
            Citation(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                quote=quote,
                char_start=abs_start,
                char_end=abs_end,
            )
        )
    return tuple(out)


__all__ = ["CitedSynthesizer", "SynthesisResult"]
