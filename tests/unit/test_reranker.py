"""Cross-encoder reranker with a fake backend — no torch, no model download."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

import pytest

from verity.retrieval.reranker import CrossEncoderReranker
from verity.types import Chunk, DocumentId, RetrievalHit, RetrieverKind


def _hit(cid_int: int, rank: int) -> RetrievalHit:
    chunk = Chunk(
        id=UUID(int=cid_int),
        document_id=DocumentId("00000000-0000-0000-0000-000000000000"),
        text=f"text-{cid_int}",
        ordinal=0,
        char_start=0,
        char_end=1,
    )
    return RetrievalHit(chunk=chunk, score=0.5, kind=RetrieverKind.FUSED, rank=rank)


class _KeywordBackend:
    """Assigns higher scores to passages containing more of the query's tokens."""

    def predict(
        self,
        sentences: Sequence[tuple[str, str]],
        *,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> list[float]:
        scores: list[float] = []
        for query, passage in sentences:
            q_terms = set(query.lower().split())
            p_terms = set(passage.lower().split())
            scores.append(float(len(q_terms & p_terms)))
        return scores


@pytest.mark.asyncio
async def test_rerank_reorders_and_truncates() -> None:
    hits = [_hit(1, 0), _hit(2, 1), _hit(3, 2)]
    # Overwrite text so the fake backend can distinguish them.
    hits = [
        RetrievalHit(
            chunk=Chunk(
                id=h.chunk.id,
                document_id=h.chunk.document_id,
                text=t,
                ordinal=h.chunk.ordinal,
                char_start=h.chunk.char_start,
                char_end=h.chunk.char_end,
            ),
            score=h.score,
            kind=h.kind,
            rank=h.rank,
        )
        for h, t in zip(hits, ["alpha beta", "gamma", "alpha beta gamma"], strict=True)
    ]

    reranker = CrossEncoderReranker(model="fake", backend=_KeywordBackend())
    out = await reranker.rerank("alpha beta gamma", hits, top_k=2)
    assert len(out) == 2
    assert out[0].chunk.text == "alpha beta gamma"  # 3 shared tokens → best
    assert out[0].kind is RetrieverKind.RERANKED
    assert [h.rank for h in out] == [0, 1]


@pytest.mark.asyncio
async def test_rerank_empty_inputs() -> None:
    reranker = CrossEncoderReranker(model="fake", backend=_KeywordBackend())
    assert await reranker.rerank("q", [], top_k=5) == []
    assert await reranker.rerank("q", [_hit(1, 0)], top_k=0) == []


@pytest.mark.asyncio
async def test_rerank_tie_break_by_input_rank() -> None:
    class _AllEqualBackend:
        def predict(
            self,
            sentences: Sequence[tuple[str, str]],
            *,
            batch_size: int = 32,
            show_progress_bar: bool = False,
            convert_to_numpy: bool = True,
        ) -> list[float]:
            return [0.42 for _ in sentences]

    hits = [_hit(1, 0), _hit(2, 1), _hit(3, 2)]
    reranker = CrossEncoderReranker(model="fake", backend=_AllEqualBackend())
    out = await reranker.rerank("q", hits, top_k=3)
    assert [h.chunk.id for h in out] == [hits[0].chunk.id, hits[1].chunk.id, hits[2].chunk.id]
