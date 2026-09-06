"""Retrieval package: protocols + hybrid assembly.

Public surface:
- Contracts: ``Fusion``, ``Reranker``, ``Retriever``, ``VectorStore``.
- Concrete: ``PgVectorStore`` (Postgres/pgvector), ``InMemoryVectorStore`` (dev/tests),
  ``ReciprocalRankFusion``, ``CrossEncoderReranker``, ``HybridRetriever``.
"""

from __future__ import annotations

from verity.retrieval.base import Fusion, Reranker, Retriever, VectorStore
from verity.retrieval.fusion import ReciprocalRankFusion
from verity.retrieval.hybrid import HybridRetriever
from verity.retrieval.memory_store import InMemoryVectorStore
from verity.retrieval.pgvector_store import PgVectorStore
from verity.retrieval.reranker import CrossEncoderReranker

__all__ = [
    "CrossEncoderReranker",
    "Fusion",
    "HybridRetriever",
    "InMemoryVectorStore",
    "PgVectorStore",
    "ReciprocalRankFusion",
    "Reranker",
    "Retriever",
    "VectorStore",
]
