"""Retrieval metrics — precision@k, recall@k, nDCG@k with binary gain.

Definitions (all against the gold set ``example.relevant_chunk_ids``):

- **precision@k** — |retrieved ∩ gold| / min(k, |retrieved|). If nothing was
  retrieved, we return 0.0 rather than raise or return NaN — same convention as
  scikit-learn, and it lets the eval harness score a broken pipeline coherently.
- **recall@k**    — |retrieved ∩ gold| / |gold|. Undefined when ``|gold| == 0``;
  the harness must therefore call these metrics only on *answerable* questions
  (the S2 spec is explicit: out-of-scope questions are scored on refusal in S3/S4,
  not on retrieval here).
- **nDCG@k**      — DCG@k / IDCG@k, with binary gain (relevant = 1). DCG uses the
  standard log2(rank+2) discount with 0-based ranks so DCG of the ideal ordering
  is Σ_{i=0..r-1} 1 / log2(i+2) where ``r = min(k, |gold|)``.

We macro-average across examples (mean of per-question scores) — micro-averaging
across a 4-question set would just amplify the single hardest question, which is not
what a scorecard should say.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from verity.types import EvalExample, RetrievalHit, RetrievalMetrics


def _hit_ids(hits: Iterable[RetrievalHit], k: int) -> list[str]:
    top = list(hits)[:k]
    return [str(h.chunk.id) for h in top]


def precision_at_k(retrieved_ids: list[str], gold: set[str], k: int) -> float:
    if k <= 0 or not retrieved_ids:
        return 0.0
    denom = min(k, len(retrieved_ids))
    return sum(1 for cid in retrieved_ids[:k] if cid in gold) / denom


def recall_at_k(retrieved_ids: list[str], gold: set[str], k: int) -> float:
    if not gold:
        raise ValueError("recall@k is undefined when the gold set is empty")
    if k <= 0 or not retrieved_ids:
        return 0.0
    return sum(1 for cid in retrieved_ids[:k] if cid in gold) / len(gold)


def ndcg_at_k(retrieved_ids: list[str], gold: set[str], k: int) -> float:
    """Binary-gain nDCG@k. Ideal DCG is computed against ``min(k, |gold|)`` ones."""
    if not gold:
        raise ValueError("nDCG@k is undefined when the gold set is empty")
    if k <= 0 or not retrieved_ids:
        return 0.0
    dcg = 0.0
    for rank, cid in enumerate(retrieved_ids[:k]):
        if cid in gold:
            dcg += 1.0 / math.log2(rank + 2)
    ideal_hits = min(k, len(gold))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


class BinaryRetrievalMetric:
    """Per-example ``RetrievalMetric`` implementation (protocol from ``eval.base``).

    Delegates to the free functions above so the same code paths are used by the
    aggregator and by unit tests exercising hand-computed cases.
    """

    def score(self, example: EvalExample, hits: list[RetrievalHit], k: int) -> RetrievalMetrics:
        gold = {str(cid) for cid in example.relevant_chunk_ids}
        retrieved = _hit_ids(hits, k)
        return RetrievalMetrics(
            precision_at_k=precision_at_k(retrieved, gold, k),
            recall_at_k=recall_at_k(retrieved, gold, k),
            ndcg_at_k=ndcg_at_k(retrieved, gold, k),
            k=k,
        )


@dataclass(frozen=True)
class PerExampleScore:
    """One row of the per-question breakdown surfaced by the CLI."""

    example_id: str
    metrics: RetrievalMetrics


def macro_average(per_example: list[PerExampleScore], k: int) -> RetrievalMetrics:
    """Macro-average per-example ``RetrievalMetrics`` into a single scorecard row.

    Zero examples → all zeros with the caller-provided ``k`` (so the caller can
    detect the empty case by inspecting the returned object, not by getting NaN).
    """
    if not per_example:
        return RetrievalMetrics(precision_at_k=0.0, recall_at_k=0.0, ndcg_at_k=0.0, k=k)
    n = float(len(per_example))
    return RetrievalMetrics(
        precision_at_k=sum(p.metrics.precision_at_k for p in per_example) / n,
        recall_at_k=sum(p.metrics.recall_at_k for p in per_example) / n,
        ndcg_at_k=sum(p.metrics.ndcg_at_k for p in per_example) / n,
        k=k,
    )


__all__ = [
    "BinaryRetrievalMetric",
    "PerExampleScore",
    "macro_average",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
]
