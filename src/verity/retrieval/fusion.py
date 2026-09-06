"""Reciprocal Rank Fusion — combine ranked lists without a common score scale.

Dense (cosine similarity) and sparse (BM25/ts_rank) produce scores on different
scales, so a straight sum would let whichever list has the wider dynamic range win
by accident. RRF sidesteps the problem by scoring on *ranks* alone:

    RRF_score(c) = Σ_i 1 / (rrf_k + rank_i(c))

where ``rank_i(c)`` is the 0-based rank of chunk ``c`` in the i-th input list, and
``rrf_k`` (default 60, following Cormack et al. 2009) dampens the head so a chunk
that appears in two lists at moderate ranks outranks one that appears in a single
list at rank 0.

We deduplicate by ``chunk.id``, add the contributions, and break ties on ``chunk.id``
so a re-run of the same input produces bit-for-bit the same output.
"""

from __future__ import annotations

from verity.config import get_settings
from verity.types import Chunk, RetrievalHit, RetrieverKind


class ReciprocalRankFusion:
    """Fusion protocol implementation. ``rrf_k`` defaults to ``config.rrf_k`` (60)."""

    def __init__(self, rrf_k: int | None = None) -> None:
        self._rrf_k = rrf_k if rrf_k is not None else get_settings().rrf_k
        if self._rrf_k < 0:
            raise ValueError(f"rrf_k must be non-negative, got {self._rrf_k}")

    def fuse(self, ranked_lists: list[list[RetrievalHit]], k: int) -> list[RetrievalHit]:
        if k <= 0:
            return []

        scores: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}

        for hits in ranked_lists:
            for rank, hit in enumerate(hits):
                cid = str(hit.chunk.id)
                # We trust the input list's rank, not hit.rank — a caller might feed us
                # a slice or a re-ordered subset and the natural interpretation is that
                # the position in the list *is* the rank at this fusion step.
                contribution = 1.0 / (self._rrf_k + rank)
                scores[cid] = scores.get(cid, 0.0) + contribution
                if cid not in chunks:
                    chunks[cid] = hit.chunk

        # Sort by (-score, chunk_id) — the chunk_id tiebreak is what makes fusion
        # bit-reproducible when two chunks accumulate the exact same RRF score.
        ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]

        return [
            RetrievalHit(
                chunk=chunks[cid],
                score=score,
                kind=RetrieverKind.FUSED,
                rank=i,
            )
            for i, (cid, score) in enumerate(ordered)
        ]


__all__ = ["ReciprocalRankFusion"]
