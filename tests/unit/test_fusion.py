"""Reciprocal Rank Fusion — determinism, dedup, tie-break, formula."""

from __future__ import annotations

from uuid import UUID

from verity.retrieval.fusion import ReciprocalRankFusion
from verity.types import Chunk, DocumentId, RetrievalHit, RetrieverKind


def _chunk(cid: UUID) -> Chunk:
    return Chunk(
        id=cid,
        document_id=DocumentId("00000000-0000-0000-0000-000000000000"),
        text=f"chunk {cid}",
        ordinal=0,
        char_start=0,
        char_end=10,
    )


def _hit(cid: UUID, rank: int, kind: RetrieverKind = RetrieverKind.DENSE) -> RetrievalHit:
    return RetrievalHit(chunk=_chunk(cid), score=1.0 / (rank + 1), kind=kind, rank=rank)


def test_rrf_score_formula() -> None:
    """Chunk A at rank 0 in both lists should score 2/(60+0) = 0.03333..."""
    a = UUID("00000000-0000-0000-0000-000000000001")
    b = UUID("00000000-0000-0000-0000-000000000002")
    dense = [_hit(a, 0), _hit(b, 1)]
    sparse = [_hit(a, 0, RetrieverKind.SPARSE), _hit(b, 1, RetrieverKind.SPARSE)]

    fused = ReciprocalRankFusion(rrf_k=60).fuse([dense, sparse], k=10)
    scores = {str(h.chunk.id): h.score for h in fused}
    assert abs(scores[str(a)] - (2.0 / 60.0)) < 1e-9
    assert abs(scores[str(b)] - (2.0 / 61.0)) < 1e-9


def test_rrf_dedup_and_kind() -> None:
    a = UUID("00000000-0000-0000-0000-000000000001")
    b = UUID("00000000-0000-0000-0000-000000000002")
    dense = [_hit(a, 0), _hit(b, 1)]
    sparse = [_hit(b, 0, RetrieverKind.SPARSE), _hit(a, 1, RetrieverKind.SPARSE)]

    fused = ReciprocalRankFusion(rrf_k=60).fuse([dense, sparse], k=10)
    ids = [str(h.chunk.id) for h in fused]
    assert len(ids) == len(set(ids)), "fusion must deduplicate by chunk id"
    assert all(h.kind is RetrieverKind.FUSED for h in fused)
    # a: 1/60 + 1/61 ; b: 1/60 + 1/61 → tie → deterministic order by chunk id.
    assert ids == sorted(ids)


def test_rrf_is_deterministic() -> None:
    a = UUID("00000000-0000-0000-0000-000000000001")
    b = UUID("00000000-0000-0000-0000-000000000002")
    c = UUID("00000000-0000-0000-0000-000000000003")
    dense = [_hit(c, 0), _hit(a, 1), _hit(b, 2)]
    sparse = [_hit(b, 0, RetrieverKind.SPARSE), _hit(c, 1, RetrieverKind.SPARSE)]
    fusion = ReciprocalRankFusion(rrf_k=60)
    run1 = [str(h.chunk.id) for h in fusion.fuse([dense, sparse], k=5)]
    run2 = [str(h.chunk.id) for h in fusion.fuse([dense, sparse], k=5)]
    assert run1 == run2


def test_rrf_truncates_to_k() -> None:
    dense = [_hit(UUID(int=i), i) for i in range(1, 6)]
    fused = ReciprocalRankFusion(rrf_k=60).fuse([dense], k=3)
    assert len(fused) == 3
    assert [h.rank for h in fused] == [0, 1, 2]


def test_rrf_zero_k_returns_empty() -> None:
    dense = [_hit(UUID(int=1), 0)]
    assert ReciprocalRankFusion(rrf_k=60).fuse([dense], k=0) == []


def test_rrf_empty_lists_return_empty() -> None:
    assert ReciprocalRankFusion(rrf_k=60).fuse([[], []], k=5) == []
