"""Stand-ins for the local models, for tests that drive the CLI in the fast lane.

The fast lane blocks the network and has no model cache, so a CLI command that builds
the real embedder cannot run there; these replace what ``verity.runtime`` builds."""

from __future__ import annotations

import pytest


class BagOfWords:
    """Embedding backend without a model: a hashed bag of words, normalised."""

    def encode(self, sentences: list[str], **_: object) -> list[list[float]]:
        rows = []
        for s in sentences:
            v = [0.0] * 32
            for tok in s.lower().split():
                v[sum(map(ord, tok)) % 32] += 1.0
            norm = sum(x * x for x in v) ** 0.5 or 1.0
            rows.append([x / norm for x in v])
        return rows

    def get_sentence_embedding_dimension(self) -> int:
        return 32


def without_local_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the CLI build an in-memory store, the bag-of-words embedder and no
    reranker."""
    import verity.runtime
    from verity.ingestion.embedder import LocalEmbedder
    from verity.retrieval import InMemoryVectorStore

    monkeypatch.setattr(
        verity.runtime,
        "build_local_components",
        lambda *, store, no_rerank=False: (
            InMemoryVectorStore(),
            LocalEmbedder(model="fake", expected_dim=32, backend=BagOfWords()),
            None,
        ),
    )
