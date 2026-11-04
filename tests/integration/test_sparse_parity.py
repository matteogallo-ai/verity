"""Sparse retrieval: in-memory BM25 against Postgres ``ts_rank`` on the shipped corpus.

The two are different scorers (BM25 with ``rank_bm25`` on ``\\w+`` tokens; ``ts_rank``
on English-stemmed lexemes), so their rankings are not identical. What this pins is
what the eval depends on: both put almost every answerable question's gold chunk in
their top 5, and they agree on the top hit for most questions. Measured on the 95
questions over 21 documents: gold in top 5 for 61/62 answerable questions with BM25
(q-041, a two-part question across two documents, is the miss) and 62/62 with
ts_rank; top-1 agreement 70/95. On the earlier 19 questions it was 11/11 and 17/19.
Gold is resolved from the evidence against the chunks ingested here, as the eval does.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.integration.db import database_url as _configured_url
from verity.db import apply_pending
from verity.eval.labels import resolve_gold
from verity.eval.retrieval_run import load_full_dataset
from verity.ingestion.embedder import LocalEmbedder
from verity.retrieval import InMemoryVectorStore, PgVectorStore
from verity.runtime import ingest_corpus
from verity.types import Chunk

pytestmark = pytest.mark.integration

_DATASET = Path("datasets/eval/questions.jsonl")
_QUESTIONS = [json.loads(line) for line in _DATASET.read_text().splitlines()]
_K = 5


@pytest.fixture(scope="module")
def database_url() -> str:
    url = _configured_url()
    apply_pending(url)
    return url


async def _top_ids(
    store: InMemoryVectorStore | PgVectorStore,
) -> tuple[dict[str, list[str]], list[Chunk]]:
    # The real embedder, although sparse search ignores embeddings: these rows are the
    # shipped corpus's, which later readers of the shared database (the CI retrieval
    # gate) expect to carry real vectors.
    embedder = LocalEmbedder()
    result = await ingest_corpus(corpus_dir=Path("datasets/corpus"), store=store, embedder=embedder)
    ranking = {
        q["id"]: [str(h.chunk.id) for h in await store.sparse_search(q["question"], _K)]
        for q in _QUESTIONS
    }
    return ranking, result.chunks


@pytest.mark.asyncio
async def test_postgres_sparse_search_matches_bm25_where_the_eval_needs_it(
    database_url: str,
) -> None:
    memory, chunks = await _top_ids(InMemoryVectorStore())
    postgres, _ = await _top_ids(PgVectorStore(database_url=database_url))

    examples, problems = resolve_gold(load_full_dataset(_DATASET), chunks)
    assert problems == []
    gold = {e.id: {str(c) for c in e.relevant_chunk_ids} for e in examples}
    answerable = [q for q in _QUESTIONS if gold[q["id"]]]
    for store_name, ranking in (("bm25", memory), ("ts_rank", postgres)):
        missed = [q["id"] for q in answerable if not gold[q["id"]] & set(ranking[q["id"]])]
        assert len(missed) <= 0.05 * len(answerable), (
            f"{store_name}: gold chunk not in top {_K} for {missed}"
        )

    without_hits = [q["id"] for q in answerable if not postgres[q["id"]]]
    assert without_hits == [], f"ts_rank found nothing for {without_hits}"

    same_top = sum(
        1 for q in _QUESTIONS if memory[q["id"]][:1] == postgres[q["id"]][:1] and memory[q["id"]]
    )
    assert same_top >= 0.7 * len(_QUESTIONS), f"top-1 agreement {same_top}/{len(_QUESTIONS)}"
