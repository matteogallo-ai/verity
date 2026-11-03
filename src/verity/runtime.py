"""Shared composition helpers so the CLI and the FastAPI layer wire the exact
same components.

Rule: the HTTP API must NOT reimplement any agent/ingestion logic. It's a serializer over the same runtime the CLI executes. Anything the
CLI does — pick embedder + reranker, ingest a corpus, build the agent with a
stub or a live routing client — lives here so ``verity ask`` and ``POST /ask``
share a single source of truth.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

import structlog

from verity.agent.agent import RagAgent, create_default_agent
from verity.agent.scripted_stub import build_stub_router
from verity.ingestion import (
    IngestionPipeline,
    IngestionResult,
    LocalEmbedder,
    StructureAwareChunker,
    create_default_parser,
)
from verity.ingestion.pipeline import discover_sources
from verity.llm.base import RoutingClient
from verity.llm.routing import create_default_routing_client, primary_client_identity
from verity.observability.base import Tracer
from verity.retrieval import (
    CrossEncoderReranker,
    HybridRetriever,
    InMemoryVectorStore,
    PgVectorStore,
)
from verity.retrieval.base import CorpusStore

log = structlog.get_logger(__name__)

AgentMode = Literal["stub", "live"]
StoreKind = Literal["pgvector", "memory"]


def build_store(kind: StoreKind) -> CorpusStore:
    """The configured store: pgvector (``VERITY_DATABASE_URL``) or in process memory."""
    if kind == "pgvector":
        return PgVectorStore()
    return InMemoryVectorStore()


def build_local_components(
    *,
    store: StoreKind,
    no_rerank: bool = False,
) -> tuple[CorpusStore, LocalEmbedder, CrossEncoderReranker | None]:
    """Instantiate the store, embedder and reranker shared by the CLI and the API."""
    embedder = LocalEmbedder()
    reranker = None if no_rerank else CrossEncoderReranker()
    return build_store(store), embedder, reranker


async def ingest_corpus(
    *,
    corpus_dir: Path,
    store: CorpusStore,
    embedder: LocalEmbedder,
) -> IngestionResult:
    """Bring the index in line with ``corpus_dir``: ingest new and edited files, skip
    unchanged ones, and remove documents whose file left the directory."""
    pipeline = IngestionPipeline(
        parser=create_default_parser(),
        chunker=StructureAwareChunker(),
        embedder=embedder,
        sink=store,
    )
    uris = discover_sources(str(corpus_dir))
    result = await pipeline.ingest(uris)
    await prune_missing(store, corpus_dir, uris)
    return result


async def prune_missing(store: CorpusStore, directory: Path, present: list[str]) -> list[str]:
    """Delete the documents ingested from files under ``directory`` that are no longer
    among ``present``; the uris removed.

    Membership is by the stored access path, so a file that still exists but failed to
    parse keeps its last good version, and uploads and URLs are never touched. A
    document ingested from another absolute path (another checkout) is not seen as
    under ``directory`` and is kept."""
    root = await asyncio.to_thread(lambda: directory.expanduser().resolve())
    kept = set(present)
    gone = [
        d
        for d in await store.list_documents()
        if d.uri not in kept and Path(d.uri).is_absolute() and Path(d.uri).is_relative_to(root)
    ]
    await store.delete_documents([d.document_id for d in gone])
    for d in gone:
        log.info("document_removed", uri=d.uri, document_id=d.document_id)
    return [d.uri for d in gone]


async def ingest_uri(
    *,
    uri: str,
    store: CorpusStore,
    embedder: LocalEmbedder,
    raw: bytes | None = None,
) -> IngestionResult:
    """Ingest a single source, optionally supplying its bytes (used by upload endpoints
    where the file is not on the local filesystem yet)."""
    parser = create_default_parser()
    chunker = StructureAwareChunker()
    doc = await parser.parse(uri, raw)
    chunks = chunker.chunk(doc)
    embedded = await embedder.embed_chunks(chunks)
    await store.replace_document(doc, embedded)
    return IngestionResult(documents=[doc], chunks=list(chunks))


def build_agent(
    *,
    retriever: HybridRetriever,
    mode: AgentMode,
    tracer: Tracer | None = None,
) -> tuple[RagAgent, str, RoutingClient]:
    """Wire the agent with either the scripted stub router or a real routing client.

    Returns ``(agent, agent_model, client)``; ``client`` is the routing client the
    agent calls, so readiness probes the providers that serve answers. ``agent_model``
    is what actually will run for every request served by this agent instance:

    - ``"stub-agent"`` for the scripted stub — the demo default when no LLM key
      is configured.
    - The live client's model id (e.g. ``"claude-sonnet-4-6"``) when a real
      routing client is wired.
    """
    if mode == "stub":
        client: RoutingClient = build_stub_router()
        agent_model = "stub-agent"
    else:
        client = create_default_routing_client()
        agent_model = primary_client_identity(client)[1] or "llm-agent"
    agent = create_default_agent(retriever=retriever, client=client, tracer=tracer)
    return agent, agent_model, client


def build_retriever(
    *,
    store: CorpusStore,
    embedder: LocalEmbedder,
    reranker: CrossEncoderReranker | None,
    tracer: Tracer | None = None,
) -> HybridRetriever:
    return HybridRetriever(
        store=store,
        embedder=embedder,
        reranker=reranker,
        tracer=tracer,
    )


__all__ = [
    "AgentMode",
    "StoreKind",
    "build_agent",
    "build_local_components",
    "build_retriever",
    "build_store",
    "ingest_corpus",
    "ingest_uri",
]
