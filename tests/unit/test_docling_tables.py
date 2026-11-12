"""Tables of a PDF or DOCX become text the retriever can find, behind VERITY_INDEX_TABLES.

Docling returns a table as a grid of cells with no ``text`` of its own, so the
flattener used to drop it: the minimum wages of the hotel-restaurant CCNT (art. 10)
were in no indexed chunk. The grids below are the ones docling reads from pages 13
and 15 of the official CCNT text (l-gav.ch). The real conversion of a table is in the
slow lane, at the end of this file.
"""

from __future__ import annotations

import io
from types import SimpleNamespace
from typing import Any

import pytest

from verity.ingestion.parsers.docling_parser import DoclingParser, _flatten


def _cell(
    text: str, *, col: bool = False, row: bool = False, start: int | None = None
) -> SimpleNamespace:
    """``start`` is the first grid column of a merged cell, repeated where it spans."""
    cell = SimpleNamespace(text=text, column_header=col, row_header=row)
    if start is not None:
        cell.start_col_offset_idx = start
    return cell


def _table(*rows: list[SimpleNamespace], caption: str = "", page: int = 13) -> SimpleNamespace:
    return SimpleNamespace(
        label="table",
        data=SimpleNamespace(grid=[list(r) for r in rows]),
        prov=[SimpleNamespace(page_no=page)],
        caption_text=lambda _doc: caption,
    )


def _heading(text: str) -> SimpleNamespace:
    return SimpleNamespace(text=text, label="section_header", level=1, prov=[])


def _para(text: str) -> SimpleNamespace:
    return SimpleNamespace(text=text, label="text", prov=[])


def _doc(*items: Any) -> SimpleNamespace:
    return SimpleNamespace(iterate_items=lambda: [(i, 0) for i in items])


# Page 13: the current minimum wages, the category in two row-header cells.
_WAGES_2026 = _table(
    [
        _cell(""),
        _cell(""),
        _cell("dès le 1.1.2026 ou saison d'été 2026", col=True),
        _cell("dès le 1.1.27 ou saison d'été 2027", col=True),
    ],
    [
        _cell("I", row=True),
        _cell("a) Collaborateurs sans apprentissage", row=True),
        _cell("3'713.-"),
        _cell("3'735.-"),
    ],
    [
        _cell("II", row=True),
        _cell("Collaborateurs ayant achevé une formation professionnelle initiale", row=True),
        _cell("4'070.-"),
        _cell("4'094.-"),
    ],
)

# Page 15: earlier wages, no row-header flag, the category in the first column.
_WAGES_HISTORY = _table(
    [
        _cell("Salaires précédents", col=True),
        _cell("01.02.25- 31.12.25", col=True),
        _cell("01.01.26- 31.12.26", col=True),
    ],
    [_cell("l a)"), _cell("3706.-"), _cell("3713.-")],
    page=15,
)


def _flat_tables(*items: Any) -> str:
    text, _ = _flatten(_doc(*items), tables=True)
    return text


def test_by_default_a_table_is_left_out_as_before() -> None:
    text, sections = _flatten(_doc(_heading("Art. 10 Salaires minimums"), _WAGES_2026))
    assert text == "Art. 10 Salaires minimums"
    assert len(sections) == 1


def test_each_row_reads_label_then_header_and_value() -> None:
    text = _flat_tables(_heading("Art. 10 Salaires minimums"), _WAGES_2026)
    assert (
        "I a) Collaborateurs sans apprentissage — dès le 1.1.2026 ou saison d'été 2026 : "
        "3'713.- ; dès le 1.1.27 ou saison d'été 2027 : 3'735.-"
    ) in text.splitlines()


def test_a_row_without_row_header_takes_its_first_cell_as_label() -> None:
    text = _flat_tables(_heading("Art. 10 Salaires minimums"), _WAGES_HISTORY)
    assert "l a) — 01.02.25- 31.12.25 : 3706.- ; 01.01.26- 31.12.26 : 3713.-" in text


def test_the_table_opens_with_its_parent_heading_and_is_one_section() -> None:
    """The chunker can label a chunk with a neighbouring heading, so the article the
    table belongs to is written into the table's own text."""
    text, sections = _flatten(_doc(_heading("Art. 10 Salaires minimums"), _WAGES_2026), tables=True)
    table = sections[1]
    block = text[table.char_start : table.char_end]
    assert block.splitlines()[0] == "Art. 10 Salaires minimums"
    assert "3'735.-" in block
    assert table.heading_path == ("Art. 10 Salaires minimums",)
    assert table.page == 13


def test_a_caption_names_the_table_rather_than_the_heading() -> None:
    captioned = _table(
        [_cell("Year", col=True), _cell("Fee", col=True)],
        [_cell("2026"), _cell("CHF 50")],
        caption="Table 2 - Fees",
    )
    text = _flat_tables(_heading("Schedule"), captioned)
    assert text.split("\n\n")[1] == "Table 2 - Fees\n2026 — Fee : CHF 50"


def test_a_table_without_column_headers_joins_its_cells() -> None:
    contents = _table(
        [_cell("Art. 5"), _cell("Temps d'essai"), _cell("chiffres 1, 2, 3")],
    )
    text = _flat_tables(_heading("II Début et fin des rapports de travail"), contents)
    assert "Art. 5 | Temps d'essai | chiffres 1, 2, 3" in text


def test_a_spanning_cell_is_written_once() -> None:
    """Docling repeats a merged cell in every grid position it spans."""
    merged = _table(
        [
            _cell("Category", col=True),
            _cell("2026", col=True, start=1),
            _cell("2026", col=True, start=1),
        ],
        [_cell("A", start=0), _cell("100", start=1), _cell("100", start=1)],
    )
    assert "A — 2026 : 100" in _flat_tables(merged).splitlines()


@pytest.mark.parametrize(
    ("rows", "line"),
    [
        ([[_cell("A"), _cell("0"), _cell("0")]], "A | 0 | 0"),
        (
            [[_cell("Cat", col=True)], [_cell("A"), _cell("0"), _cell("0")]],
            "A — 0 ; 0",
        ),
    ],
)
def test_equal_values_in_separate_cells_are_both_kept(
    rows: list[list[SimpleNamespace]], line: str
) -> None:
    assert line in _flat_tables(_table(*rows)).splitlines()


def test_a_header_row_below_data_heads_only_the_rows_after_it() -> None:
    sections = _table(
        [_cell("Cat", col=True), _cell("2026", col=True)],
        [_cell("A"), _cell("1")],
        [_cell("Section B", col=True, start=0), _cell("Section B", col=True, start=0)],
        [_cell("B"), _cell("2")],
    )
    assert _flat_tables(sections).splitlines() == ["A — 2026 : 1", "Section B", "B — 2026 : 2"]


def test_a_label_flagged_as_both_headers_is_kept() -> None:
    table = _table(
        [_cell("Cat", col=True), _cell("2026", col=True)],
        [_cell("I", col=True, row=True), _cell("1")],
    )
    assert "I — 2026 : 1" in _flat_tables(table).splitlines()


def test_an_empty_label_does_not_promote_a_value() -> None:
    table = _table(
        [_cell("Cat", col=True), _cell("2026", col=True), _cell("2027", col=True)],
        [_cell(""), _cell("1"), _cell("2")],
    )
    assert "2026 : 1 ; 2027 : 2" in _flat_tables(table).splitlines()


def test_a_table_with_no_text_adds_nothing() -> None:
    empty = _table([_cell(""), _cell("")], [_cell(" "), _cell("")])
    assert _flat_tables(_para("Body."), empty) == "Body."


def test_a_table_before_any_heading_has_no_title_line() -> None:
    text = _flat_tables(_WAGES_HISTORY)
    assert text.splitlines()[0].startswith("l a) — ")


class _Converter:
    def __init__(self, document: Any) -> None:
        self._document = document

    def convert(self, _path: str) -> SimpleNamespace:
        return SimpleNamespace(document=self._document)


@pytest.mark.asyncio
async def test_the_parser_follows_the_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    from verity.config import get_settings

    converter = _Converter(_doc(_heading("Art. 10 Salaires minimums"), _WAGES_2026))
    off = await DoclingParser(converter).parse("upload://ccnt.docx", raw=b"PK")  # type: ignore[arg-type]
    monkeypatch.setenv("VERITY_INDEX_TABLES", "true")
    get_settings.cache_clear()
    on = await DoclingParser(converter).parse("upload://ccnt.docx", raw=b"PK")  # type: ignore[arg-type]
    assert "3'713.-" not in off.text
    assert "3'713.-" in on.text


@pytest.mark.slow
@pytest.mark.asyncio
async def test_a_real_docx_table_is_indexed() -> None:
    from docx import Document as DocxDocument

    document = DocxDocument()
    document.add_heading("Art. 10 Salaires minimums", level=1)
    table = document.add_table(rows=2, cols=2)
    table.rows[0].cells[0].text = "Catégorie"
    table.rows[0].cells[1].text = "dès le 1.1.2026"
    table.rows[1].cells[0].text = "I a) Collaborateurs sans apprentissage"
    table.rows[1].cells[1].text = "3'713.-"
    buffer = io.BytesIO()
    document.save(buffer)

    parsed = await DoclingParser(index_tables=True).parse(
        "upload://ccnt.docx", raw=buffer.getvalue()
    )
    assert "3'713.-" in parsed.text
    assert "Art. 10 Salaires minimums\n" in parsed.text
