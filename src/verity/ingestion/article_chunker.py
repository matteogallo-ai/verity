"""Article chunker for legal texts (``VERITY_CHUNKING=articles``).

A collective labour agreement is read and cited by article, but the headings docling
reports are often not the articles: a table row or a sub-title becomes the heading, and
the structure chunker then labels a passage with it. On the landscapers' CCT, art. 9.1
ended a chunk labelled "D) Jardiniers-grimpeurs", and a citation from it named no
article. This chunker cuts the text at each article heading and labels every passage
with the article's number and title as the text writes them.

What counts as an article heading is decided on the line alone, then against the
numbering so far:

- a line of at most 160 characters reading "Art. N", "Article N", "Article premier" or
  "§ N", optionally followed by a title, and not by a reference ("Art. 10 al. 4 ...");
- whose number is higher than the current article's, or 1, where a new text begins
  (the decree that extends an agreement numbers its own articles, then the agreement
  starts again at 1). A repeated number is a table caption repeating its article;
  a lower one is a list or a reference, and stays in the current passage.

An "Annexe" heading opens an annex. Inside it, a short line without digits or final
punctuation ("Genève", "Tous les cantons") titles the tables under it, and starts a new
passage: a cantonal wage scale is then not read as the scale of every canton before it.

A passage stays a verbatim slice of the document (the offset invariant of
``verity.ingestion.chunker``), so the label lives in ``chunk.section`` and the number and
title in ``chunk.metadata``. An article over the word budget is split by paragraph, each
part keeping the label; a heading with almost no text under it (a list of articles, a
table of contents) joins the passage before it. A document with neither an article nor
an annex heading is chunked by the structure chunker, unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from verity.ingestion.chunker import StructureAwareChunker, _find_paragraphs, _Paragraph
from verity.ingestion.ids import chunk_id_for
from verity.ingestion.structure import DocumentStructure, Section
from verity.types import Chunk, Document

_DEFAULT_MAX_WORDS = 500
# A passage smaller than this joins the next one in its article or annex, so a heading
# is not a passage of its own.
_MIN_PIECE_WORDS = 20
# The shortest provision an article heading needs below it to be an article.
_MIN_ARTICLE_BODY_WORDS = 5
_MAX_HEADING_CHARS = 160
# An annex's own articles start right below its heading; an article 1 further down is
# the start of another text appended to the annex (an agreement after its last annex).
_ANNEX_ARTICLES_WITHIN_PARAGRAPHS = 3

_ARTICLE_RE = re.compile(
    r"(?:Art(?:icle)?\.?|ART(?:ICLE)?\.?|§)[ \t]*(?P<num>premier|1er|1re|\d+)"
    r"(?P<suffix>bis|ter|quater|[a-z](?![^\W\d_]))?(?P<rest>.*)"
)
_REFERENCE_RE = re.compile(
    r"(?:al|alin[ée]a|ch|chiffre|let|lettre|lit|litt|ss|et|à|ou|CO|CC|CCT|LTN|LTr|OLT)\b"
)
# A colon or a final period makes a line of prose ("Annexe au présent contrat : ..."),
# not the heading of an annex.
_ANNEX_RE = re.compile(r"(?:ANNEXE|Annexe)(?:[ \t]+[^:]*[^:.\s])?")
_SEPARATORS = " \t\N{EN DASH}\N{EM DASH}-:|."


@dataclass(frozen=True)
class _Heading:
    pos: int
    label: str
    metadata: dict[str, str]
    is_annex: bool


@dataclass
class _Unit:
    start: int
    end: int
    label: str | None
    metadata: dict[str, str] = field(default_factory=dict)
    is_annex: bool = False


def _article(line: str) -> tuple[str, int, str] | None:
    """(number as written, sort key, title) of an article heading line, else None."""
    if len(line) > _MAX_HEADING_CHARS:
        return None
    match = _ARTICLE_RE.fullmatch(line)
    if match is None:
        return None
    rest = match["rest"]
    if rest and rest[0] not in _SEPARATORS:
        return None
    if rest.startswith(".") and rest[1:2].isdigit():
        return None  # "Art. 12.1": a sub-provision named in the text
    title = rest.strip(_SEPARATORS)
    if title.startswith(("(", ",", ";")) or _REFERENCE_RE.match(title):
        return None
    # A title starts with a capital or a figure and ends without a sentence's
    # punctuation; "Art. 15 de la présente convention s'applique." names an article.
    if title and (title[0].islower() or rest.rstrip()[-1] in ".:;"):
        return None
    digits = "1" if match["num"] in ("premier", "1er", "1re") else match["num"]
    suffix = match["suffix"] or ""
    return digits + suffix, int(digits) * 2 + (1 if suffix else 0), title


def _headings(text: str) -> list[_Heading]:
    headings: list[_Heading] = []
    current: int | None = None
    annex: str | None = None
    paragraphs_since_annex = 0
    pos = 0
    previous_blank = True
    for raw in text.splitlines(keepends=True):
        line = raw.strip()
        if line and previous_blank:
            paragraphs_since_annex += 1
        if len(line) <= _MAX_HEADING_CHARS and _ANNEX_RE.fullmatch(line):
            annex, current, paragraphs_since_annex = line, None, 0
            headings.append(_Heading(pos, line, {"annex": line}, is_annex=True))
        elif (found := _article(line)) is not None:
            number, key, title = found
            if current is None or key > current or (key == 2 and current > 2):
                if (
                    key == 2
                    and annex
                    and paragraphs_since_annex > _ANNEX_ARTICLES_WITHIN_PARAGRAPHS
                ):
                    annex = None
                current = key
                metadata = {"article": number, "article_title": title}
                if annex:
                    metadata["annex"] = annex
                label = f"{annex} > {line}" if annex else line
                headings.append(_Heading(pos, label, metadata, is_annex=False))
        previous_blank = not line
        pos += len(raw)
    return headings


def _units(text: str, headings: list[_Heading]) -> list[_Unit]:
    """Contiguous spans of ``text``, one per article or annex; together they cover it."""
    units: list[_Unit] = []
    if text[: headings[0].pos].strip():
        units.append(_Unit(0, headings[0].pos, None))
    # An annex heading with little below it ("Annexe VI", "Caution") opens the
    # article that follows rather than standing as a passage of its own.
    opened_at: int | None = None
    for heading, following in zip(headings, [*headings[1:], None], strict=True):
        end = following.pos if following else len(text)
        span = text[heading.pos : end]
        if heading.is_annex and following and len(span.split()) < _MIN_PIECE_WORDS:
            opened_at = heading.pos if opened_at is None else opened_at
            continue
        if units and not heading.is_annex and not _has_provision(text, heading.pos, end):
            if opened_at is None:
                units[-1].end = end
            continue
        start = heading.pos if opened_at is None else opened_at
        units.append(_Unit(start, end, heading.label, heading.metadata, heading.is_annex))
        opened_at = None
    if opened_at is not None:
        # The annex heads opened nothing: no article with a provision followed them.
        if units:
            units[-1].end = len(text)
        else:
            units.append(_Unit(opened_at, len(text), None))
    return units


def _has_provision(text: str, start: int, end: int) -> bool:
    """Whether an article heading has text of its own below it: a sentence or a table.

    A list of articles (an annex naming the ones that do not apply, a table of
    contents) puts nothing below a heading but the next heading or a group title.
    """
    newline = text.find("\n", start, end)
    if newline < 0:
        return False
    body = Section(char_start=newline, char_end=end)
    for paragraph in _find_paragraphs(text, body):
        block = text[paragraph.char_start : paragraph.char_end]
        if paragraph.word_count >= _MIN_ARTICLE_BODY_WORDS and (
            "\n" in block or block[-1] in ".;:!?"
        ):
            return True
    return False


def _is_title(text: str, paragraph: _Paragraph) -> bool:
    line = text[paragraph.char_start : paragraph.char_end]
    return (
        "\n" not in line
        and paragraph.word_count <= 5
        and not any(c.isdigit() for c in line)
        and "…" not in line
        and ".." not in line
        and line[-1] not in ".,;:"
    )


class ArticleChunker:
    """One passage per article (or annex part), labelled with its real number."""

    def __init__(self, *, max_words: int = _DEFAULT_MAX_WORDS) -> None:
        if max_words <= 0:
            raise ValueError("max_words must be positive")
        self._max = max_words

    def chunk(self, document: Document) -> list[Chunk]:
        headings = _headings(document.text)
        if not headings:
            return StructureAwareChunker().chunk(document)
        sections = DocumentStructure.from_metadata(document.metadata).sections
        chunks: list[Chunk] = []
        for unit in _units(document.text, headings):
            for start, end, label in self._pieces(document.text, unit):
                chunks.append(
                    Chunk(
                        id=chunk_id_for(document.id, len(chunks), start, end),
                        document_id=document.id,
                        text=document.text[start:end],
                        ordinal=len(chunks),
                        char_start=start,
                        char_end=end,
                        section=label or _heading_at(sections, start),
                        page=_page_at(sections, start),
                        metadata=dict(unit.metadata),
                    )
                )
        return chunks

    def _pieces(self, text: str, unit: _Unit) -> list[tuple[int, int, str | None]]:
        paragraphs = _find_paragraphs(text, Section(char_start=unit.start, char_end=unit.end))
        pieces: list[tuple[int, int, str | None]] = []
        last_words = 0
        current: list[_Paragraph] = []
        words = 0
        path: list[str] = []
        after_title = False

        def label() -> str | None:
            return " > ".join([unit.label, *path]) if unit.label else None

        def flush() -> None:
            nonlocal current, words, last_words
            if current:
                pieces.append((current[0].char_start, current[-1].char_end, label()))
                last_words = words
            current, words = [], 0

        for i, paragraph in enumerate(paragraphs):
            if unit.is_annex and i > 0 and _is_title(text, paragraph):
                if not after_title:
                    # Text under a title is that title's, however short; text before
                    # any title (the annex's own heading) opens the first titled part.
                    if words >= _MIN_PIECE_WORDS or path:
                        flush()
                    path = []
                path.append(text[paragraph.char_start : paragraph.char_end])
                after_title = True
            else:
                if current and words + paragraph.word_count > self._max:
                    flush()
                after_title = False
            current.append(paragraph)
            words += paragraph.word_count
        # A short tail joins the part before it only when it is under the same title
        # and the two fit the budget.
        if (
            current
            and pieces
            and words < _MIN_PIECE_WORDS
            and pieces[-1][2] == label()
            and last_words + words <= self._max
        ):
            start, _, previous = pieces[-1]
            pieces[-1] = (start, current[-1].char_end, previous)
        else:
            flush()
        return pieces


def _heading_at(sections: tuple[Section, ...], pos: int) -> str | None:
    for section in sections:
        if section.char_start <= pos < section.char_end and section.heading_path:
            return " > ".join(section.heading_path)
    return None


def _page_at(sections: tuple[Section, ...], pos: int) -> int | None:
    for section in sections:
        if section.char_start <= pos < section.char_end:
            return section.page
    return None


__all__ = ["ArticleChunker"]
