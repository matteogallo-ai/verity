"""Ingestion: parse → chunk → embed. Public surface exported from here.

Import a parser factory or the pipeline runner from this package; the concrete
classes live in submodules but the protocol contracts (verity.ingestion.base) are
what everything outside this package should depend on.
"""

from __future__ import annotations

from verity.ingestion.base import Chunker, Embedder, Parser
from verity.ingestion.chunker import StructureAwareChunker
from verity.ingestion.embedder import LocalEmbedder
from verity.ingestion.ids import chunk_id_for, document_id_for_uri, identity_for
from verity.ingestion.parsers import create_default_parser, detect_source_type
from verity.ingestion.pipeline import IngestionPipeline, IngestionResult

__all__ = [
    "Chunker",
    "Embedder",
    "IngestionPipeline",
    "IngestionResult",
    "LocalEmbedder",
    "Parser",
    "StructureAwareChunker",
    "chunk_id_for",
    "create_default_parser",
    "detect_source_type",
    "document_id_for_uri",
    "identity_for",
]
