"""Property-based tests of the Markdown, text and docling-flattening parsers.

Uploaded bytes are hostile input: any byte string must parse into a Document (decoding
replaces invalid UTF-8) whose sections partition its text, so the chunker's offsets
stay valid. The real docling conversion loads layout models and takes seconds per
file, so it stays in the slow lane (test_docling_parser.py); here its flattening step
runs on generated item trees.
"""

from __future__ import annotations

import asyncio
import itertools
from types import SimpleNamespace

from hypothesis import given
from hypothesis import strategies as st

from verity.ingestion.ids import document_id_for_uri
from verity.ingestion.parsers.docling_parser import _HEADING_LABELS, _flatten
from verity.ingestion.parsers.markdown import MarkdownParser
from verity.ingestion.parsers.text import TextParser
from verity.ingestion.structure import DocumentStructure
from verity.types import Document

# Bytes that look like Markdown often enough to hit headings, plus arbitrary bytes.
_MARKDOWN_BYTES = st.one_of(
    st.binary(max_size=300),
    st.lists(
        st.sampled_from([b"# ", b"## ", b"###### ", b"####### ", b"\n", b"\r\n", b"\r", b" "])
        | st.text(max_size=12).map(str.encode),
        max_size=30,
    ).map(b"".join),
)


def _normalised(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")


def _assert_sections_partition(document: Document) -> None:
    sections = DocumentStructure.from_metadata(document.metadata).sections
    assert sections, "a parsed document always carries at least one section"
    assert sections[0].char_start == 0
    assert sections[-1].char_end == len(document.text)
    for previous, current in itertools.pairwise(sections):
        assert previous.char_end == current.char_start
    assert all(s.char_start <= s.char_end for s in sections)


@given(_MARKDOWN_BYTES)
def test_markdown_parses_any_bytes_into_partitioning_sections(raw: bytes) -> None:
    document = asyncio.run(MarkdownParser().parse("file:///corpus/generated.md", raw))
    assert document.text == _normalised(raw)
    assert "\r" not in document.text
    assert document.id == document_id_for_uri("file:///corpus/generated.md")
    _assert_sections_partition(document)
    for section in DocumentStructure.from_metadata(document.metadata).sections:
        if section.heading_path:
            assert document.text[section.char_start] == "#"
            assert all(h and h == h.strip() for h in section.heading_path)
            heading_line = document.text[section.char_start : section.char_end].split("\n")[0]
            assert section.heading_path[-1] in heading_line


def test_a_hash_without_heading_text_is_not_a_heading() -> None:
    """Found by the partition property: ``#`` followed by tabs became a heading named
    ``""``, and ``#`` alone on a line took the next paragraph as its heading text."""
    for raw in (b"#\t\t", b"#\n\nBody paragraph."):
        document = asyncio.run(MarkdownParser().parse("file:///corpus/generated.md", raw))
        sections = DocumentStructure.from_metadata(document.metadata).sections
        assert [s.heading_path for s in sections] == [()]
        assert document.title == "generated"


def test_unicode_spaces_after_the_separator_still_make_a_heading() -> None:
    """Only the separator after ``#`` must be a space or tab (CommonMark)."""
    raw = "# \u00a0Title".encode()
    document = asyncio.run(MarkdownParser().parse("file:///corpus/generated.md", raw))
    assert document.title == "Title"


@given(_MARKDOWN_BYTES)
def test_markdown_title_is_the_first_heading_or_the_file_stem(raw: bytes) -> None:
    document = asyncio.run(MarkdownParser().parse("file:///corpus/generated.md", raw))
    sections = DocumentStructure.from_metadata(document.metadata).sections
    headed = [s for s in sections if s.heading_path]
    assert document.title == (headed[0].heading_path[-1] if headed else "generated")


@given(st.binary(max_size=300))
def test_text_parses_any_bytes_into_one_section(raw: bytes) -> None:
    document = asyncio.run(TextParser().parse("file:///corpus/notes.txt", raw))
    assert document.text == _normalised(raw)
    assert document.title == "notes"
    _assert_sections_partition(document)
    assert len(DocumentStructure.from_metadata(document.metadata).sections) == 1


_LABELS = st.sampled_from([*sorted(_HEADING_LABELS), "text", "list_item", "table"])
_ITEMS = st.lists(
    st.builds(
        SimpleNamespace,
        text=st.one_of(st.none(), st.text(max_size=20)),
        label=_LABELS,
        level=st.one_of(st.none(), st.integers(-1, 4)),
        prov=st.lists(st.builds(SimpleNamespace, page_no=st.integers(1, 9)), max_size=1),
    ),
    max_size=12,
)


@given(_ITEMS)
def test_docling_flatten_spans_point_at_each_item_text(items: list[SimpleNamespace]) -> None:
    text, sections = _flatten(SimpleNamespace(iterate_items=lambda: [(i, 0) for i in items]))
    kept = [i for i in items if isinstance(i.text, str) and i.text.strip()]
    if not kept:
        assert text == ""
        assert [(s.char_start, s.char_end) for s in sections] == [(0, 0)]
        return
    assert len(sections) == len(kept)
    for item, section in zip(kept, sections, strict=True):
        assert text[section.char_start : section.char_end] == item.text.strip()
        expected_page = item.prov[0].page_no if item.prov else None
        assert section.page == expected_page
        if item.label in _HEADING_LABELS:
            assert section.heading_path[-1] == item.text.strip()
    assert text == "\n\n".join(i.text.strip() for i in kept)
