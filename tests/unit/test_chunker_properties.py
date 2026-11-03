"""Property-based tests of the chunker: offsets, coverage, ordering, no empty chunks.

Documents are generated as paragraphs of words separated by blank lines, optionally
under Markdown headings, with Unicode whitespace mixed in, and chunked with small word
budgets so that packing, overlap and the small-tail merge all occur on short inputs.
"""

from __future__ import annotations

import itertools

from hypothesis import given
from hypothesis import strategies as st

from verity.ingestion.chunker import StructureAwareChunker
from verity.ingestion.ids import chunk_id_for
from verity.ingestion.parsers.markdown import _sections_from_headings
from verity.ingestion.structure import DocumentStructure
from verity.types import Document, SourceType

# Whitespace a parser can hand the chunker: ASCII, no-break space, ideographic space,
# vertical tab, form feed.
_SPACES = st.sampled_from([" ", "  ", "\t", "\u00a0", "\u3000", "\x0b", "\x0c"])
_WORD = st.text(alphabet="abcdefghij0123456789.", min_size=1, max_size=8)


@st.composite
def _paragraph(draw: st.DrawFn) -> str:
    words = draw(st.lists(_WORD, min_size=0, max_size=12))
    pieces = [draw(_SPACES) if i and draw(st.booleans()) else " " for i in range(len(words))]
    body = "".join(p + w for p, w in zip(pieces, words, strict=True)).lstrip(" ")
    # A paragraph made only of whitespace is a valid input the chunker must skip.
    return body + draw(st.sampled_from(["", " ", "\u00a0", "\x0b"]))


@st.composite
def _document(draw: st.DrawFn) -> Document:
    blocks: list[str] = []
    for _ in range(draw(st.integers(0, 10))):
        if draw(st.integers(0, 4)) == 0:
            blocks.append("#" * draw(st.integers(1, 3)) + " " + draw(_WORD))
        else:
            blocks.append(draw(_paragraph()))
    separators = [draw(st.sampled_from(["\n\n", "\n\n\n", "\n"])) for _ in blocks]
    text = "".join(b + s for b, s in zip(blocks, separators, strict=True))
    structure = DocumentStructure(sections=_sections_from_headings(text))
    return Document(
        source_type=SourceType.MARKDOWN,
        uri="file:///generated.md",
        text=text,
        metadata=structure.to_metadata(),
    )


@st.composite
def _chunker(draw: st.DrawFn) -> StructureAwareChunker:
    target = draw(st.integers(1, 30))
    return StructureAwareChunker(
        target_words=target,
        max_words=target + draw(st.integers(0, 20)),
        min_words=draw(st.integers(0, 10)),
        overlap_paragraphs=draw(st.integers(0, 2)),
    )


@given(_document(), _chunker())
def test_every_chunk_is_the_exact_slice_its_offsets_name(
    document: Document, chunker: StructureAwareChunker
) -> None:
    for chunk in chunker.chunk(document):
        assert document.text[chunk.char_start : chunk.char_end] == chunk.text
        assert chunk.document_id == document.id
        assert chunk.id == chunk_id_for(
            document.id, chunk.ordinal, chunk.char_start, chunk.char_end
        )


@given(_document(), _chunker())
def test_no_chunk_is_empty_or_whitespace(
    document: Document, chunker: StructureAwareChunker
) -> None:
    for chunk in chunker.chunk(document):
        assert chunk.text.strip(), f"whitespace-only chunk {chunk.text!r}"


@given(_document(), _chunker())
def test_every_non_whitespace_character_is_in_some_chunk(
    document: Document, chunker: StructureAwareChunker
) -> None:
    covered = [False] * len(document.text)
    for chunk in chunker.chunk(document):
        for i in range(chunk.char_start, chunk.char_end):
            covered[i] = True
    missing = [i for i, ch in enumerate(document.text) if not ch.isspace() and not covered[i]]
    assert not missing, f"characters {missing[:5]} are in no chunk"


@given(_document(), _chunker())
def test_chunks_are_ordered_and_each_adds_new_text(
    document: Document, chunker: StructureAwareChunker
) -> None:
    chunks = chunker.chunk(document)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert len({c.id for c in chunks}) == len(chunks)
    for previous, current in itertools.pairwise(chunks):
        assert current.char_start >= previous.char_start
        # Overlap repeats a paragraph; a chunk that ends where its predecessor ends
        # repeats it entirely and adds nothing retrievable.
        assert current.char_end > previous.char_end, (
            f"chunk {current.ordinal} {current.text!r} adds nothing to chunk "
            f"{previous.ordinal} {previous.text!r}"
        )


@given(_document())
def test_chunking_is_deterministic(document: Document) -> None:
    assert StructureAwareChunker().chunk(document) == StructureAwareChunker().chunk(document)


def test_a_paragraph_over_the_target_is_emitted_once() -> None:
    """Found by the ordering property: the overlap paragraph was re-emitted alone."""
    text = " ".join(["word"] * 450)
    document = Document(source_type=SourceType.TXT, uri="file:///long.txt", text=text)
    chunks = StructureAwareChunker().chunk(document)
    assert [(c.char_start, c.char_end) for c in chunks] == [(0, len(text))]


def test_a_whitespace_only_paragraph_is_not_a_chunk() -> None:
    """Found by the no-empty-chunk property (thorough profile): a no-break space."""
    document = Document(source_type=SourceType.TXT, uri="file:///nbsp.txt", text="\u00a0\n\n")
    assert StructureAwareChunker().chunk(document) == []
