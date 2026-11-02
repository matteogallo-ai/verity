"""Sparse retrieval: in-memory BM25 against Postgres ``ts_rank`` on the shipped corpus.

The two are different scorers (BM25 with ``rank_bm25`` on ``\\w+`` tokens; ``ts_rank``
on English-stemmed lexemes), so their rankings are not identical. What this pins is
what the eval depends on: both put every answerable question's gold chunk in their
top 5, and they agree on the top hit for most questions. Measured on 19 questions:
top-1 agreement 17/19, mean Jaccard@5 0.59, gold in top 5 11/11 for both.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from verity.db import apply_pending
from verity.ingestion.embedder import LocalEmbedder
from verity.retrieval import InMemoryVectorStore, PgVectorStore
from verity.runtime import ingest_corpus

pytestmark = pytest.mark.integration

_QUESTIONS = [
    json.loads(line) for line in Path("datasets/eval/questions.jsonl").read_text().splitlines()
]
_K = 5


@pytest.fixture(scope="module")
def database_url() -> str:
    url = os.environ.get("VERITY_DATABASE_URL", "postgresql://verity:verity@localhost:5432/verity")
    apply_pending(url)
    return url


async def _top_ids(store: InMemoryVectorStore | PgVectorStore) -> dict[str, list[str]]:
    # The real embedder, although sparse search ignores embeddings: these rows are the
    # shipped corpus's, which later readers of the shared database (the CI retrieval
    # gate) expect to carry real vectors.
    embedder = LocalEmbedder()
    await ingest_corpus(corpus_dir=Path("datasets/corpus"), store=store, embedder=embedder)
    return {
        q["id"]: [str(h.chunk.id) for h in await store.sparse_search(q["question"], _K)]
        for q in _QUESTIONS
    }


@pytest.mark.asyncio
async def test_postgres_sparse_search_matches_bm25_where_the_eval_needs_it(
    database_url: str,
) -> None:
    memory = await _top_ids(InMemoryVectorStore())
    postgres = await _top_ids(PgVectorStore(database_url=database_url))

    answerable = [q for q in _QUESTIONS if q["relevant_chunk_ids"]]
    for store_name, ranking in (("bm25", memory), ("ts_rank", postgres)):
        missed = [
            q["id"] for q in answerable if not set(q["relevant_chunk_ids"]) & set(ranking[q["id"]])
        ]
        assert missed == [], f"{store_name}: gold chunk not in top {_K} for {missed}"

    without_hits = [q["id"] for q in answerable if not postgres[q["id"]]]
    assert without_hits == [], f"ts_rank found nothing for {without_hits}"

    same_top = sum(
        1 for q in _QUESTIONS if memory[q["id"]][:1] == postgres[q["id"]][:1] and memory[q["id"]]
    )
    assert same_top >= 0.8 * len(_QUESTIONS), f"top-1 agreement {same_top}/{len(_QUESTIONS)}"
