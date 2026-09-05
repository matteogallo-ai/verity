"""Local embedder backed by sentence-transformers.

Two design choices worth stating:

- **Local by default.** The default model (BAAI/bge-small-en-v1.5, 384-dim) runs on
  CPU with no API key. This is what keeps CI free and the eval loop deterministic.
  The weights are downloaded on the *first* run — see the README notice.

- **Injectable encoder.** ``LocalEmbedder`` accepts an ``encoder_factory`` so unit
  tests can substitute a fake encoder without pulling torch or a model into the
  test process. The default factory imports sentence-transformers lazily.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Protocol, cast

from verity.config import get_settings
from verity.ingestion.base import Embedder
from verity.types import Chunk, EmbeddedChunk


class EmbeddingBackend(Protocol):
    """Structural interface for the underlying encoder.

    ``sentence_transformers.SentenceTransformer`` satisfies this shape, as does any
    fake encoder that returns an iterable of numeric vectors. Return type is ``Any``
    on purpose — real backends return numpy arrays or torch tensors and we iterate
    row-by-row in the caller.
    """

    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int = ...,
        normalize_embeddings: bool = ...,
        show_progress_bar: bool = ...,
        convert_to_numpy: bool = ...,
    ) -> Any: ...

    def get_sentence_embedding_dimension(self) -> int | None: ...


def _default_backend_factory(model_name: str) -> EmbeddingBackend:
    """Lazily import sentence-transformers so the module is importable without torch."""
    from sentence_transformers import SentenceTransformer

    # SentenceTransformer's stubs are wider than our narrow protocol; the cast is the
    # single boundary where we accept its shape as an EmbeddingBackend.
    return cast(EmbeddingBackend, SentenceTransformer(model_name))


class LocalEmbedder(Embedder):
    """A batched, async-friendly wrapper around a sentence-transformers backend.

    ``embed`` is async by contract; the underlying encoder is synchronous, so we hop
    to a worker thread via :func:`asyncio.to_thread` to avoid blocking the event
    loop. Batching is applied on the caller's list — no re-batching gymnastics.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        expected_dim: int | None = None,
        batch_size: int = 32,
        normalize: bool = True,
        backend: EmbeddingBackend | None = None,
        backend_factory: Callable[[str], EmbeddingBackend] = _default_backend_factory,
    ) -> None:
        settings = get_settings()
        self._model = model or settings.embedding_model
        self._expected_dim = expected_dim if expected_dim is not None else settings.embedding_dim
        self._batch_size = batch_size
        self._normalize = normalize
        self._backend: EmbeddingBackend | None = backend
        self._backend_factory = backend_factory
        self._dim_verified = False

    @property
    def model(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        return self._expected_dim

    def _get_backend(self) -> EmbeddingBackend:
        if self._backend is None:
            self._backend = self._backend_factory(self._model)
        if not self._dim_verified:
            actual = self._backend.get_sentence_embedding_dimension()
            if actual is None:
                raise RuntimeError(
                    f"Embedding model {self._model!r} did not report an embedding dimension."
                )
            if actual != self._expected_dim:
                raise RuntimeError(
                    f"Embedding model {self._model!r} produces dim={actual}, "
                    f"but config.embedding_dim={self._expected_dim}. "
                    "Adjust VERITY_EMBEDDING_DIM or VERITY_EMBEDDING_MODEL to match."
                )
            self._dim_verified = True
        return self._backend

    def _encode_sync(self, texts: list[str]) -> list[tuple[float, ...]]:
        backend = self._get_backend()
        raw = backend.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        rows: Iterable[Iterable[float]] = raw  # backends return numpy ndarray or list-of-lists
        result: list[tuple[float, ...]] = []
        for row in rows:
            vec = tuple(float(x) for x in row)
            if len(vec) != self._expected_dim:
                raise RuntimeError(
                    f"Encoder returned vector of length {len(vec)} "
                    f"but expected {self._expected_dim}"
                )
            result.append(vec)
        return result

    async def embed(self, texts: list[str]) -> list[tuple[float, ...]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._encode_sync, texts)

    async def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        if not chunks:
            return []
        vectors = await self.embed([c.text for c in chunks])
        return [
            EmbeddedChunk(chunk=c, embedding=v, model=self._model)
            for c, v in zip(chunks, vectors, strict=True)
        ]


__all__ = ["EmbeddingBackend", "LocalEmbedder"]
