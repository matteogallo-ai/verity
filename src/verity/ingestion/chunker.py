"""Structure-aware chunker with a load-bearing offset invariant.

Two rules govern this module:

1. **Semantic boundaries first.** The chunker never splits inside a paragraph if it
   can help it and never crosses a section heading if it can avoid it. Sections come
   from :class:`DocumentStructure` (populated by the parser); paragraphs are found by
   splitting on blank lines within a section.

2. **The offset invariant.** For every produced chunk ``c``,
   ``document.text[c.char_start:c.char_end] == c.text`` must hold. This is asserted
   by a test that runs across the whole example corpus and is what keeps citations
   pointing at exact source passages.

The token budget is approximated (a real tokeniser would tie us to the LLM's model
choice, which is out of scope at ingestion time) — we use a whitespace word count and
a conservative 1.3 words-per-token ratio. The resulting chunks land around 400-600
tokens for typical prose (400-600 words), which sits comfortably in every embedder's
context window.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from verity.ingestion.ids import chunk_id_for
from verity.ingestion.structure import DocumentStructure, Section
from verity.types import Chunk, Document

log = structlog.get_logger(__name__)

# Approximate token budget expressed in *words* (see module docstring).
_DEFAULT_TARGET_WORDS = 400
_DEFAULT_MAX_WORDS = 500
_DEFAULT_MIN_WORDS = 40


@dataclass(frozen=True)
class _Paragraph:
    """One paragraph inside a section, indexed into ``document.text``."""

    char_start: int
    char_end: int
    word_count: int


def _find_paragraphs(text: str, section: Section) -> tuple[_Paragraph, ...]:
    """Split ``text[section.char_start:section.char_end]`` on blank lines.

    Blank-line splits keep chunk boundaries on structural whitespace, and the
    returned offsets index the *whole document text* — not the section slice — so
    they can flow straight into ``Chunk`` without recomputation.
    """
    span_start = section.char_start
    span_end = section.char_end
    body = text[span_start:span_end]

    paragraphs: list[_Paragraph] = []
    cursor = 0
    n = len(body)
    while cursor < n:
        while cursor < n and body[cursor] in {"\n", "\r"} and body[cursor : cursor + 2] != "\n\n":
            cursor += 1
        # skip pure blank runs
        while cursor < n and body.startswith("\n\n", cursor):
            cursor += 2
        while cursor < n and body[cursor] == "\n":
            cursor += 1
        if cursor >= n:
            break
        start = cursor
        # Advance until the next paragraph break (two consecutive newlines).
        while cursor < n:
            if body.startswith("\n\n", cursor):
                break
            cursor += 1
        end = cursor
        # Trim trailing whitespace *inside* the span (not the leading whitespace of
        # the paragraph — leaving it in would break the invariant on re-slicing).
        while end > start and body[end - 1] in {" ", "\t", "\n"}:
            end -= 1
        if end <= start:
            continue
        word_count = len(body[start:end].split())
        paragraphs.append(
            _Paragraph(
                char_start=span_start + start,
                char_end=span_start + end,
                word_count=word_count,
            )
        )

    if not paragraphs and body.strip():
        # A section with no paragraph breaks (e.g. a one-liner heading only).
        stripped_start = span_start + (len(body) - len(body.lstrip()))
        stripped_end = span_end - (len(body) - len(body.rstrip()))
        if stripped_end > stripped_start:
            wc = len(text[stripped_start:stripped_end].split())
            paragraphs.append(
                _Paragraph(char_start=stripped_start, char_end=stripped_end, word_count=wc)
            )
    return tuple(paragraphs)


class StructureAwareChunker:
    """Packs paragraphs into token-budgeted chunks along section boundaries.

    Constructor parameters use word counts as a proxy for tokens (see module docstring).
    ``overlap_paragraphs`` (default 1) carries the last paragraph of chunk *n* into
    chunk *n+1* so that a claim straddling a chunk boundary still has enough context
    to be retrievable — a common practical improvement over strict non-overlap.
    """

    def __init__(
        self,
        *,
        target_words: int = _DEFAULT_TARGET_WORDS,
        max_words: int = _DEFAULT_MAX_WORDS,
        min_words: int = _DEFAULT_MIN_WORDS,
        overlap_paragraphs: int = 1,
    ) -> None:
        if target_words <= 0 or max_words < target_words or min_words < 0:
            raise ValueError("chunker word budgets must satisfy 0 < target ≤ max and min ≥ 0")
        self._target = target_words
        self._max = max_words
        self._min = min_words
        self._overlap = overlap_paragraphs

    def chunk(self, document: Document) -> list[Chunk]:
        structure = DocumentStructure.from_metadata(document.metadata)
        sections = structure.sections or (Section(char_start=0, char_end=len(document.text)),)

        chunks: list[Chunk] = []
        ordinal = 0

        for section in sections:
            paragraphs = _find_paragraphs(document.text, section)
            if not paragraphs:
                continue

            buffer: list[_Paragraph] = []
            buffer_words = 0

            for paragraph in paragraphs:
                # A single paragraph over the max budget still gets emitted as one chunk
                # rather than being split mid-sentence — chunkers that split arbitrary
                # sentences produce citations that read like garbage. Big paragraphs are
                # rare in practice; the retriever downstream handles length just fine.
                if buffer and buffer_words + paragraph.word_count > self._max:
                    chunks.append(self._emit(document, section, buffer, ordinal))
                    ordinal += 1
                    buffer = buffer[-self._overlap :] if self._overlap > 0 else []
                    buffer_words = sum(p.word_count for p in buffer)
                buffer.append(paragraph)
                buffer_words += paragraph.word_count

                if buffer_words >= self._target:
                    chunks.append(self._emit(document, section, buffer, ordinal))
                    ordinal += 1
                    buffer = buffer[-self._overlap :] if self._overlap > 0 else []
                    buffer_words = sum(p.word_count for p in buffer)

            if buffer and buffer_words >= self._min:
                chunks.append(self._emit(document, section, buffer, ordinal))
                ordinal += 1
            elif buffer and chunks:
                # A small trailing buffer is merged into the previous chunk rather than
                # emitted as an underweight chunk, which would hurt retrieval quality.
                last = chunks[-1]
                new_end = buffer[-1].char_end
                merged_text = document.text[last.char_start : new_end]
                chunks[-1] = Chunk(
                    id=chunk_id_for(document.id, last.ordinal, last.char_start, new_end),
                    document_id=document.id,
                    text=merged_text,
                    ordinal=last.ordinal,
                    char_start=last.char_start,
                    char_end=new_end,
                    section=last.section,
                    page=last.page,
                    metadata=last.metadata,
                )
            elif buffer:
                # No previous chunk to merge with — emit even if under min_words so we
                # never drop content silently.
                chunks.append(self._emit(document, section, buffer, ordinal))
                ordinal += 1

        return chunks

    def _emit(
        self,
        document: Document,
        section: Section,
        buffer: list[_Paragraph],
        ordinal: int,
    ) -> Chunk:
        cs = buffer[0].char_start
        ce = buffer[-1].char_end
        text = document.text[cs:ce]
        heading = " > ".join(section.heading_path) if section.heading_path else None
        word_count = sum(p.word_count for p in buffer)
        if word_count > self._max:
            # Fires when a single paragraph exceeds max_words (the packer never crosses
            # the ceiling on its own). Downstream embedders will silently truncate a
            # too-long input to their context window — we want that visible.
            log.warning(
                "chunk_over_budget",
                document_uri=document.uri,
                ordinal=ordinal,
                word_count=word_count,
                max_words=self._max,
                section=heading,
            )
        return Chunk(
            id=chunk_id_for(document.id, ordinal, cs, ce),
            document_id=document.id,
            text=text,
            ordinal=ordinal,
            char_start=cs,
            char_end=ce,
            section=heading,
            page=section.page,
        )


__all__ = ["StructureAwareChunker"]
