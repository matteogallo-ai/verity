"""Integrity check: eval labels resolve to real chunks with expected anchor text.

This test is the guard against the "portable ids" regression: if
``document_id_for_uri`` ever starts deriving ids from machine-specific paths again,
the ids committed in ``datasets/eval/questions.jsonl`` will stop resolving to any
chunk produced by the ingestion pipeline and this test fails loudly.

Marked ``@pytest.mark.integration`` so the fast unit lane stays free (the pipeline
still touches docling/torch through the default parser factory); the integration lane
runs it on every push.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from verity.ingestion import StructureAwareChunker, create_default_parser
from verity.ingestion.pipeline import discover_sources
from verity.types import Chunk

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = REPO_ROOT / "datasets" / "corpus"
DATASET_PATH = REPO_ROOT / "datasets" / "eval" / "questions.jsonl"

# For each answerable question, the substring(s) at least one of its labeled chunks
# must contain. These are the human-readable anchors that make the label meaningful.
_EXPECTED_ANCHORS: dict[str, tuple[str, ...]] = {
    "q-001": ("thirty (30) days",),
    "q-002": ("data-processing", "Acme Corporation"),
    "q-003": ("$482.0 million",),
    "q-005": ("competitor",),
}


def _load_corpus_chunks() -> dict[str, Chunk]:
    parser = create_default_parser()
    chunker = StructureAwareChunker()

    async def run() -> list[Chunk]:
        uris = discover_sources(str(CORPUS_DIR))
        chunks: list[Chunk] = []
        for uri in uris:
            doc = await parser.parse(uri)
            chunks.extend(chunker.chunk(doc))
        return chunks

    chunks = asyncio.run(run())
    return {str(c.id): c for c in chunks}


def _load_dataset() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with DATASET_PATH.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            records.append(json.loads(raw))
    return records


def test_eval_labels_resolve_to_existing_chunks() -> None:
    by_id = _load_corpus_chunks()
    assert by_id, "corpus produced no chunks — check datasets/corpus/"

    for record in _load_dataset():
        qid = str(record["id"])
        answerable = bool(record["answerable"])
        chunk_ids = record.get("relevant_chunk_ids") or []
        assert isinstance(chunk_ids, list)

        if not answerable:
            assert chunk_ids == [], (
                f"{qid}: unanswerable question must keep relevant_chunk_ids == [], "
                f"got {chunk_ids!r}"
            )
            continue

        assert chunk_ids, f"{qid}: answerable question has no relevant_chunk_ids"

        for cid in chunk_ids:
            assert cid in by_id, (
                f"{qid}: relevant_chunk_id {cid!r} does not match any chunk produced "
                "by the current corpus + chunker. Regenerate with "
                "`uv run python scripts/build_eval_labels.py` and commit the result."
            )

        anchors = _EXPECTED_ANCHORS.get(qid)
        if anchors is None:
            continue
        matched = [cid for cid in chunk_ids if all(a in by_id[cid].text for a in anchors)]
        assert matched, (
            f"{qid}: none of the labeled chunks contains all expected anchors "
            f"{anchors!r}. Chunk texts were:\n"
            + "\n---\n".join(by_id[cid].text[:200] for cid in chunk_ids)
        )
