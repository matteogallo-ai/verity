"""The committed docling fixture stands in for every corpus PDF and DOCX.

The fixture is the labelling machine's parse; the committed gold chunk ids are pinned
to it (``tests/unit/test_eval_labels.py``). A live parse is not compared to it byte
for byte: docling's layout model parses the corpus PDFs differently on other CPUs (on
the GitHub runner, numbered headings became list items), so that comparison failed on
hardware, not on a change. ``tests/integration/test_eval_labels_integrity.py`` checks,
on the live parse, the part the eval depends on: every evidence quote is in a chunk.
Regenerate after a corpus or docling change with ``uv run python -m tests.docling_fixture``.
"""

from __future__ import annotations

from tests import docling_fixture


def test_fixture_covers_every_pdf_and_docx_in_the_corpus() -> None:
    assert set(docling_fixture.load()) == {p.name for p in docling_fixture.corpus_files()}
