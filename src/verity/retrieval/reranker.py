"""Cross-encoder reranker — local, offline, CI-safe.

A cross-encoder scores (query, passage) pairs jointly and consistently beats the
bi-encoder embeddings that produced them for *tightening* a small candidate set. The
default model (``cross-encoder/ms-marco-MiniLM-L-6-v2``) runs on CPU in tens of
milliseconds per pair, weighs ~90 MB, and needs no API key — which is what lets the
eval harness run in CI without secrets.

The backend is injectable (same pattern as :class:`LocalEmbedder`). The unit lane
never loads a real model; the cast at the boundary is where sentence-transformers'
richer overload set is narrowed to our two-method protocol.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import Any, Protocol, cast

from verity.config import get_settings
from verity.retrieval.base import Reranker
from verity.types import RetrievalHit, RetrieverKind


class RerankerBackend(Protocol):
    """Structural interface for a cross-encoder that scores (query, passage) pairs.

    Return type is ``Any`` because real backends return numpy arrays or torch tensors
    — the caller iterates row-by-row.
    """

    def predict(
        self,
        sentences: Sequence[tuple[str, str]],
        *,
        batch_size: int = ...,
        show_progress_bar: bool = ...,
        convert_to_numpy: bool = ...,
    ) -> Any: ...


def _default_backend_factory(model_name: str) -> RerankerBackend:
    """Import sentence-transformers only when a real rerank is executed."""
    from sentence_transformers import CrossEncoder

    return cast(RerankerBackend, CrossEncoder(model_name))


class CrossEncoderReranker(Reranker):
    """Local cross-encoder reranker.

    ``rerank`` re-scores the given hits with the underlying model, sorts descending
    by that score, truncates to ``top_k``, and re-assigns 0-based ranks. The output
    score is the raw cross-encoder logit — larger = more relevant — documented on
    :class:`~verity.types.RetrieverKind` (``RERANKED`` semantics).
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        batch_size: int = 32,
        backend: RerankerBackend | None = None,
        backend_factory: Callable[[str], RerankerBackend] = _default_backend_factory,
    ) -> None:
        self._model = model or get_settings().reranker_model
        self._batch_size = batch_size
        self._backend: RerankerBackend | None = backend
        self._backend_factory = backend_factory

    @property
    def model(self) -> str:
        return self._model

    def _get_backend(self) -> RerankerBackend:
        if self._backend is None:
            self._backend = self._backend_factory(self._model)
        return self._backend

    def _predict_sync(self, pairs: list[tuple[str, str]]) -> list[float]:
        backend = self._get_backend()
        raw = backend.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        # numpy ndarray → list[float]; a fake backend returning a plain list also works.
        return [float(x) for x in raw]

    async def rerank(self, query: str, hits: list[RetrievalHit], top_k: int) -> list[RetrievalHit]:
        if not hits or top_k <= 0:
            return []
        pairs: list[tuple[str, str]] = [(query, h.chunk.text) for h in hits]
        scores = await asyncio.to_thread(self._predict_sync, pairs)

        # Sort desc by score. Tie-break on the input rank so a rerank that produces
        # exactly-equal scores keeps the fusion order — reproducibility again.
        scored = sorted(
            zip(scores, hits, strict=True),
            key=lambda sh: (-sh[0], sh[1].rank),
        )[:top_k]

        return [
            RetrievalHit(
                chunk=h.chunk,
                score=float(score),
                kind=RetrieverKind.RERANKED,
                rank=i,
            )
            for i, (score, h) in enumerate(scored)
        ]


__all__ = ["CrossEncoderReranker", "RerankerBackend"]
