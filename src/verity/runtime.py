"""Shared composition helpers so the CLI and the FastAPI layer wire the exact
same components.

Rule of iron for S6: the HTTP API must NOT reimplement any agent/ingestion
logic. It's a serializer over the same runtime the CLI executes. Anything the
CLI does — pick embedder + reranker, ingest a corpus, build the agent with a
stub or a live routing client — lives here so ``verity ask`` and ``POST /ask``
share a single source of truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

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
from verity.llm.routing import create_default_routing_client
from verity.observability.base import Tracer
from verity.retrieval import (
    CrossEncoderReranker,
    HybridRetriever,
    InMemoryVectorStore,
)

AgentMode = Literal["stub", "live"]


def build_local_components(
    *,
    no_rerank: bool = False,
) -> tuple[InMemoryVectorStore, LocalEmbedder, CrossEncoderReranker | None]:
    """Instantiate the local components used by both CLI and API in in-memory mode."""
    store = InMemoryVectorStore()
    embedder = LocalEmbedder()
    reranker = None if no_rerank else CrossEncoderReranker()
    return store, embedder, reranker


async def ingest_corpus(
    *,
    corpus_dir: Path,
    store: InMemoryVectorStore,
    embedder: LocalEmbedder,
) -> IngestionResult:
    """Run parse → chunk → embed → upsert on every source in ``corpus_dir``."""
    pipeline = IngestionPipeline(
        parser=create_default_parser(),
        chunker=StructureAwareChunker(),
        embedder=embedder,
        sink=store,
    )
    uris = discover_sources(str(corpus_dir))
    return await pipeline.ingest(uris)


async def ingest_uri(
    *,
    uri: str,
    store: InMemoryVectorStore,
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
    await store.upsert_documents([doc])
    await store.upsert(embedded)
    return IngestionResult(documents=[doc], chunks=list(chunks))


def build_agent(
    *,
    retriever: HybridRetriever,
    mode: AgentMode,
    tracer: Tracer | None = None,
) -> tuple[RagAgent, str]:
    """Wire the agent with either the scripted stub router or a real routing client.

    Returns ``(agent, agent_model)``. ``agent_model`` is what actually will run
    for every request served by this agent instance:

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
        agent_model = _first_client_model(client) or "llm-agent"
    agent = create_default_agent(retriever=retriever, client=client, tracer=tracer)
    return agent, agent_model


def build_retriever(
    *,
    store: InMemoryVectorStore,
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


def _first_client_model(client: RoutingClient) -> str | None:
    clients = getattr(client, "clients", None)
    if not clients:
        return None
    model = getattr(clients[0], "model", None)
    return model if isinstance(model, str) else None


__all__ = [
    "AgentMode",
    "build_agent",
    "build_local_components",
    "build_retriever",
    "ingest_corpus",
    "ingest_uri",
]
