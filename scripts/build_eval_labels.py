"""Populate ``relevant_chunk_ids`` in the eval dataset from the example corpus.

Runs the ingestion pipeline in dry-run mode (parse + chunk only, no DB writes and no
embedding model download), then matches each *answerable* question to the chunks that
should be retrieved by looking for a set of question-specific *anchor phrases* in the
chunk text. The anchors are hand-curated below so the mapping is reviewable and stable.

The matching is deliberately dumb keyword-lookup, not semantic search: we are producing
gold labels for a retrieval-quality benchmark — using the retriever itself to label the
retriever would defeat the exercise.

Unanswerable questions (q-004, q-006) keep an empty ``relevant_chunk_ids`` list — they
are, by design, out of scope for this corpus (see ADR 0003).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from verity.ingestion import StructureAwareChunker, create_default_parser
from verity.ingestion.pipeline import IngestionPipeline, discover_sources
from verity.types import Chunk


class _NullEmbedder:
    """Chunker-only stand-in so we never touch the sentence-transformers backend."""

    @property
    def model(self) -> str:
        return "null"

    @property
    def dim(self) -> int:
        return 0

    async def embed(self, texts: list[str]) -> list[tuple[float, ...]]:
        return [() for _ in texts]

    async def embed_chunks(self, chunks: list[Chunk]) -> list:  # type: ignore[type-arg]
        return []


# Anchor phrases per answerable question. A chunk is labeled relevant if it contains
# ALL anchors in one of the alternative anchor sets (list of sets). Multiple sets can
# match multiple chunks — used for multi-hop questions whose answer spans two sections.
_ANCHORS: dict[str, list[set[str]]] = {
    # MSA
    "q-001": [{"terminate", "thirty (30) days"}],
    "q-002": [{"data-processing", "Acme Corporation"}, {"one million", "$1,000,000"}],
    "q-005": [{"assignment", "competitor"}, {"assign this Agreement", "consent"}],
    # Financial summary
    "q-003": [{"$482.0 million", "18.4%"}],
    # SLA
    "q-007": [{"99.9%"}, {"below 95.0%", "50%"}],
    "q-010": [{"Priority 1", "one (1) hour"}],
    "q-013": [{"scheduled maintenance", "forty-eight (48) hours"}],
    # DPA
    "q-008": [{"Frankfurt"}, {"Standard Contractual Clauses"}],
    "q-009": [{"ninety (90) days"}],
    # Employee handbook
    "q-011": [{"twenty-five (25) days"}],
    "q-012": [{"two (2) days per week"}],
}


REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO_ROOT / "datasets" / "corpus"
DATASET_PATH = REPO_ROOT / "datasets" / "eval" / "questions.jsonl"


async def _load_chunks() -> list[Chunk]:
    parser = create_default_parser()
    chunker = StructureAwareChunker()
    pipeline = IngestionPipeline(parser=parser, chunker=chunker, embedder=_NullEmbedder())
    uris = discover_sources(str(CORPUS_DIR))
    result = await pipeline.ingest(uris)
    return result.chunks


def _match(chunks: list[Chunk], anchor_sets: list[set[str]]) -> list[str]:
    matches: list[str] = []
    for chunk in chunks:
        text = chunk.text
        for anchors in anchor_sets:
            if all(a in text for a in anchors):
                matches.append(str(chunk.id))
                break
    return matches


def main() -> int:
    chunks = asyncio.run(_load_chunks())
    if not chunks:
        raise SystemExit("No chunks produced from corpus — check datasets/corpus/")

    lines: list[dict] = []  # type: ignore[type-arg]
    with DATASET_PATH.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            record = json.loads(raw)
            qid = record["id"]
            if record.get("answerable") and qid in _ANCHORS:
                ids = _match(chunks, _ANCHORS[qid])
                record["relevant_chunk_ids"] = ids
                if not ids:
                    print(f"[warn] {qid}: no chunk matched — check anchor phrases")
                else:
                    print(f"[ok]   {qid}: {len(ids)} chunk(s) labeled")
            else:
                record["relevant_chunk_ids"] = []
                print(f"[skip] {qid}: unanswerable / no anchors — kept empty")
            lines.append(record)

    with DATASET_PATH.open("w", encoding="utf-8") as fh:
        for record in lines:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\nWrote {DATASET_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
