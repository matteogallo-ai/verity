"""Process-scoped runtime shared by every HTTP request.

One :class:`InMemoryVectorStore`, one :class:`LocalEmbedder`, one
:class:`CrossEncoderReranker`, one :class:`RagAgent`. The eval corpus is
preloaded on startup so the demo works with zero uploads; ``POST /ingest``
adds on top.

Why a singleton: the demo lives inside one FastAPI process. Rebuilding the
embedder + reranker on every request would waste seconds of model loading
each time and defeat the S5 cold-start work.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from verity.agent.agent import RagAgent
from verity.api.models import Provenance

if TYPE_CHECKING:  # pragma: no cover — types only
    from verity.types import Answer
from verity.config import get_settings
from verity.ingestion import LocalEmbedder
from verity.observability.base import Tracer
from verity.observability.tracer import NoOpTracer
from verity.retrieval import CrossEncoderReranker, InMemoryVectorStore
from verity.retrieval.hybrid import HybridRetriever
from verity.runtime import (
    AgentMode,
    build_agent,
    build_local_components,
    build_retriever,
    ingest_corpus,
    ingest_uri,
)

_UPLOAD_URI_PREFIX = "upload://"
_FILENAME_SANITISE_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class IngestedSource:
    """Bookkeeping for the /status endpoint — one entry per ingested Document."""

    uri: str
    title: str | None
    document_id: str
    n_chunks: int
    ingested_at_ms: float


class AppRuntime:
    """Owns the shared retriever + agent + ingestion history for the API layer.

    Both :func:`api.router.ingest_endpoint` and :func:`api.router.ask_endpoint`
    call methods on this instance — they never build their own agent or store.
    That is what makes the CLI (``verity ingest`` / ``verity ask``) and the API
    behave identically: one composition function (:mod:`verity.runtime`) builds
    the components, both surfaces reuse the same instance.
    """

    def __init__(
        self,
        *,
        mode: AgentMode,
        corpus_dir: Path,
        tracer: Tracer | None = None,
    ) -> None:
        self._mode: AgentMode = mode
        self._corpus_dir = corpus_dir
        self._tracer: Tracer = tracer or NoOpTracer()
        self._lock = asyncio.Lock()
        self._preloaded = False
        self._sources: dict[str, IngestedSource] = {}

        self.store: InMemoryVectorStore
        self.embedder: LocalEmbedder
        self.reranker: CrossEncoderReranker | None
        self.retriever: HybridRetriever
        self.agent: RagAgent
        self.agent_model: str

        self.store, self.embedder, self.reranker = build_local_components(no_rerank=False)
        self.retriever = build_retriever(
            store=self.store,
            embedder=self.embedder,
            reranker=self.reranker,
            tracer=self._tracer,
        )
        self.agent, self.agent_model = build_agent(
            retriever=self.retriever, mode=self._mode, tracer=self._tracer
        )

    async def preload_corpus(self) -> None:
        """Ingest ``corpus_dir`` once. Safe to call more than once."""
        async with self._lock:
            if self._preloaded:
                return
            if not self._corpus_dir.exists():
                self._preloaded = True
                return
            result = await ingest_corpus(
                corpus_dir=self._corpus_dir,
                store=self.store,
                embedder=self.embedder,
            )
            for doc in result.documents:
                self._sources[str(doc.id)] = IngestedSource(
                    uri=doc.uri,
                    title=doc.title,
                    document_id=str(doc.id),
                    n_chunks=sum(1 for c in result.chunks if c.document_id == doc.id),
                    ingested_at_ms=time.time() * 1000,
                )
            self._preloaded = True

    async def ingest_bytes(self, *, filename: str, content: bytes) -> IngestedSource:
        """Ingest a client-uploaded file.

        The URI is a **pure content address** — ``upload://<sha256><ext>`` —
        with the client-supplied filename NOT part of the URI. Same bytes
        under a different name therefore always resolve to the same
        ``document_id`` (``uuid5(NAMESPACE_URL, uri)``); re-uploads are
        idempotent by construction. The filename is preserved separately as
        the ``title`` for display, and the extension (parsed off the sanitised
        name) is what drives the ``SourceType`` dispatch.
        """
        safe_name = _sanitise_filename(filename)
        digest = hashlib.sha256(content).hexdigest()
        ext = _extension_for(safe_name)
        access_uri = f"{_UPLOAD_URI_PREFIX}{digest}{ext}"
        result = await ingest_uri(
            uri=access_uri,
            store=self.store,
            embedder=self.embedder,
            raw=content,
        )
        assert result.documents, "ingest_uri produced no document"
        doc = result.documents[0]
        record = IngestedSource(
            uri=doc.uri,
            title=safe_name,
            document_id=str(doc.id),
            n_chunks=len(result.chunks),
            ingested_at_ms=time.time() * 1000,
        )
        self._sources[str(doc.id)] = record
        return record

    async def ask(self, question: str) -> Answer:
        return await self.agent.answer(question)

    def sources(self) -> Sequence[IngestedSource]:
        return list(self._sources.values())

    def provenance(self) -> Provenance:
        """Fallback provenance derived from the boot-time configuration.

        Used only by ``/status`` (and by any call path that hasn't produced
        an :class:`Answer` yet). Never a substitute for
        :meth:`provenance_for` — call sites that HAVE an Answer must use that
        one so a live-mode routing fallback is reported honestly per request.
        """
        settings = get_settings()
        return Provenance(
            agent_model=self.agent_model,
            is_stub=(self._mode == "stub"),
            embedding_model=settings.embedding_model,
            reranker_model=settings.reranker_model,
        )

    def provenance_for(self, answer: Answer) -> Provenance:
        """Per-request provenance stamped from :attr:`Answer.provider_used`.

        This is the load-bearing surface: what the UI badge, ``AskResponse``,
        and any downstream metric consumer sees. ``Answer.provider_used`` is
        populated by :class:`RagAgent` from the synthesizer's ``Completion``,
        with the router's first-configured provider stamped as a fallback so
        even a graceful short-circuit refusal (e.g. mid-flight LLM error)
        still carries the honest routing decision.

        When ``provider_used`` is genuinely ``None`` — the caller built an
        Answer manually, no routing decision available — we surface
        ``agent_model="unknown"`` explicitly. We deliberately do **not**
        silently fall back to the boot-time :meth:`provenance` here: a live
        deployment with a stub-side fallback that misreports as "live" is
        exactly the class of dishonest label this fix closes.
        """
        from verity.types import Provider

        settings = get_settings()
        provider = getattr(answer, "provider_used", None)
        model = getattr(answer, "model_used", None)
        if provider is None:
            return Provenance(
                agent_model="unknown",
                is_stub=(self._mode == "stub"),
                embedding_model=settings.embedding_model,
                reranker_model=settings.reranker_model,
            )
        is_stub = provider is Provider.LOCAL
        return Provenance(
            agent_model="stub-agent" if is_stub else (model or "llm-agent"),
            is_stub=is_stub,
            embedding_model=settings.embedding_model,
            reranker_model=settings.reranker_model,
        )


def _sanitise_filename(name: str) -> str:
    stripped = name.strip().split("/")[-1].split("\\")[-1] or "upload"
    return _FILENAME_SANITISE_RE.sub("_", stripped) or "upload"


def _extension_for(name: str) -> str:
    """Recover a parser-dispatchable extension from a sanitised filename.

    Falls back to ``.txt`` so bare uploads without an extension still parse.
    """
    lowered = name.lower()
    for ext in (".pdf", ".docx", ".md", ".markdown", ".txt"):
        if lowered.endswith(ext):
            return ext
    return ".txt"


def default_agent_mode() -> AgentMode:
    """Stub by default when no LLM key is configured — matches the CLI guard."""
    settings = get_settings()
    if settings.anthropic_api_key or settings.openai_api_key:
        return "live"
    return "stub"


__all__ = ["AppRuntime", "IngestedSource", "default_agent_mode"]
