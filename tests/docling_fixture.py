"""Docling's output for the corpus PDF and DOCX files, committed as a fixture.

The fast test lane runs without network and without docling's models (a fresh CI
runner has neither the layout models nor RapidOCR's). Its tests seed retrievers from
``datasets/corpus``, so they read the parsed text of the PDF/DOCX documents from
``tests/fixtures/docling_corpus.json`` instead of running docling.
``tests/unit/test_docling_corpus_fixture.py`` (slow lane, real docling) fails when the
fixture no longer matches a live parse.

    uv run python -m tests.docling_fixture   # rewrite the fixture after a corpus change
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from verity.ingestion.ids import document_id_for_uri
from verity.types import Document

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS = REPO_ROOT / "datasets" / "corpus"
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "docling_corpus.json"
SUFFIXES = (".pdf", ".docx")


def corpus_files() -> list[Path]:
    return sorted(p for p in CORPUS.iterdir() if p.suffix in SUFFIXES)


def _entry(doc: Document) -> dict[str, object]:
    return {
        "source_type": doc.source_type.value,
        "title": doc.title,
        "text": doc.text,
        "metadata": doc.metadata,
    }


async def parse_live() -> dict[str, dict[str, object]]:
    """Parse every corpus PDF/DOCX with the real docling parser."""
    from verity.ingestion.parsers.docling_parser import DoclingParser

    parser = DoclingParser()
    # Raw bytes: the unit-test conftest serves corpus paths from this fixture and only
    # lets uploads through to docling, so a live parse must look like an upload.
    return {p.name: _entry(await parser.parse(str(p), raw=p.read_bytes())) for p in corpus_files()}


def load() -> dict[str, dict[str, object]]:
    data: dict[str, dict[str, object]] = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return data


def document(uri: str, entry: dict[str, object]) -> Document:
    """The Document docling would return for ``uri``; ``ingested_at`` is the only
    field that differs from a live parse."""
    return Document.model_validate({**entry, "id": document_id_for_uri(uri), "uri": uri})


def render(data: dict[str, dict[str, object]]) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(render(asyncio.run(parse_live())), encoding="utf-8")
    print(f"wrote {FIXTURE}")
