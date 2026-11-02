"""Fill ``relevant_chunk_ids`` in the eval dataset from each question's ``evidence``.

Every answerable question lists, in ``evidence``, verbatim quotes from the corpus that
together answer it. A chunk is gold for the question when it contains one of the
quotes (whitespace-normalised). The corpus is parsed and chunked in dry-run mode: no
database, no embedding model. Matching is plain substring lookup on purpose: using
the retriever to label the retriever would defeat the benchmark.

Fails (exit 1) when an answerable question has no evidence, a quote matches no chunk,
or an unanswerable question carries evidence.

    uv run python scripts/build_eval_labels.py           # rewrite the labels
    uv run python scripts/build_eval_labels.py --check   # exit 1 if they would change
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

from verity.ingestion import StructureAwareChunker, create_default_parser
from verity.ingestion.pipeline import IngestionPipeline, discover_sources
from verity.types import Chunk

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO_ROOT / "datasets" / "corpus"
DATASET_PATH = REPO_ROOT / "datasets" / "eval" / "questions.jsonl"


class _NullEmbedder:
    """Chunker-only stand-in so labelling never loads the sentence-transformers model."""

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


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


async def _load_chunks() -> list[Chunk]:
    pipeline = IngestionPipeline(
        parser=create_default_parser(), chunker=StructureAwareChunker(), embedder=_NullEmbedder()
    )
    result = await pipeline.ingest(discover_sources(str(CORPUS_DIR)))
    return result.chunks


def label(records: list[dict], chunks: list[Chunk]) -> tuple[list[dict], list[str]]:  # type: ignore[type-arg]
    """The records with ``relevant_chunk_ids`` filled, and every labelling problem."""
    texts = [(str(c.id), _normalise(c.text)) for c in chunks]
    problems: list[str] = []
    out: list[dict] = []  # type: ignore[type-arg]
    for record in records:
        qid, evidence = record["id"], record.get("evidence", [])
        ids: list[str] = []
        if record["answerable"] and not evidence:
            problems.append(f"{qid}: answerable but has no evidence")
        if not record["answerable"] and evidence:
            problems.append(f"{qid}: unanswerable but has evidence")
        for quote in evidence:
            hits = [cid for cid, text in texts if _normalise(quote) in text]
            if not hits:
                problems.append(f"{qid}: no chunk contains {quote!r}")
            ids.extend(h for h in hits if h not in ids)
        out.append({**record, "relevant_chunk_ids": ids})
    return out, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if labels would change")
    args = parser.parse_args(argv)

    records = [json.loads(line) for line in DATASET_PATH.read_text().splitlines() if line.strip()]
    labelled, problems = label(records, asyncio.run(_load_chunks()))
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        return 1
    rendered = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in labelled)
    if args.check:
        if rendered != DATASET_PATH.read_text():
            print("labels are stale: run scripts/build_eval_labels.py", file=sys.stderr)
            return 1
        print(f"labels up to date ({len(labelled)} questions)")
        return 0
    DATASET_PATH.write_text(rendered)
    print(f"labelled {len(labelled)} questions -> {DATASET_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
