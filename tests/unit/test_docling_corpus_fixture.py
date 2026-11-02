"""The committed docling fixture is what docling produces for the corpus today.

Slow lane: runs the real docling models. On failure, regenerate with
``uv run python -m tests.docling_fixture`` and review the diff.
"""

from __future__ import annotations

import pytest

from tests import docling_fixture

pytestmark = pytest.mark.slow


@pytest.mark.asyncio
async def test_fixture_matches_a_live_docling_parse() -> None:
    live = await docling_fixture.parse_live()
    assert docling_fixture.render(live) == docling_fixture.FIXTURE.read_text(encoding="utf-8")


def test_fixture_covers_every_pdf_and_docx_in_the_corpus() -> None:
    assert set(docling_fixture.load()) == {p.name for p in docling_fixture.corpus_files()}
