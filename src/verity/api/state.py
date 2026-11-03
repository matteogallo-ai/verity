"""Process-scoped runtime shared by every HTTP request.

One store (pgvector by default, ``VERITY_STORE=memory`` for no database), one
:class:`LocalEmbedder`, one :class:`CrossEncoderReranker`, one :class:`RagAgent`.
At startup the eval corpus is ingested (an idempotent upsert) and both models are
loaded, in the background, so the demo works with zero uploads and the first
answer pays no load; ``/ready`` is 503 until that is done. ``POST /ingest`` adds on
top. ``/status`` lists what the
store holds, so with pgvector it includes uploads from earlier processes.

Why a singleton: the demo lives inside one FastAPI process. Rebuilding the
embedder + reranker on every request would waste seconds of model loading
each time; one process-wide instance pays the model load once.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import structlog

from verity.agent.agent import RagAgent
from verity.api.models import Provenance

if TYPE_CHECKING:  # pragma: no cover — types only
    from verity.types import Answer
from verity.config import get_settings
from verity.ingestion import LocalEmbedder
from verity.ingestion.ids import document_id_for_uri
from verity.llm.base import LLMClient
from verity.llm.routing import MultiProviderRoutingClient
from verity.observability.base import Tracer
from verity.observability.tracer import NoOpTracer
from verity.retrieval import CrossEncoderReranker
from verity.retrieval.base import CorpusStore, DocumentSummary
from verity.retrieval.hybrid import HybridRetriever
from verity.runtime import (
    AgentMode,
    StoreKind,
    build_agent,
    build_local_components,
    build_retriever,
    ingest_corpus,
    ingest_uri,
)

_log = structlog.get_logger("verity.api.state")
_UPLOAD_URI_PREFIX = "upload://"
# Any question will do: retrieving it loads the query embedder and the reranker.
_PRELOAD_QUERY = "preload"
PreloadState = Literal["loading", "ok", "failed"]
_FILENAME_SANITISE_RE = re.compile(r"[^A-Za-z0-9._-]+")


class CorpusFullError(Exception):
    """The shared corpus already holds the configured maximum of documents."""

    def __init__(self, limit: int) -> None:
        super().__init__(f"corpus holds the maximum of {limit} documents")
        self.limit = limit


@dataclass
class IngestedSource:
    """What ``POST /ingest`` reports about the document it just stored."""

    uri: str
    title: str | None
    document_id: str
    n_chunks: int


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
        store: StoreKind,
        tracer: Tracer | None = None,
    ) -> None:
        self._mode: AgentMode = mode
        self._corpus_dir = corpus_dir
        self._tracer: Tracer = tracer or NoOpTracer()
        self._lock = asyncio.Lock()
        self._ingest_lock = asyncio.Lock()
        self._preloaded = False
        self.preload_state: PreloadState = "loading"
        self._preload_task: asyncio.Task[None] | None = None

        self.store: CorpusStore
        self.embedder: LocalEmbedder
        self.reranker: CrossEncoderReranker | None
        self.retriever: HybridRetriever
        self.agent: RagAgent
        self.agent_model: str

        self.store, self.embedder, self.reranker = build_local_components(
            store=store, no_rerank=False
        )
        self.retriever = build_retriever(
            store=self.store,
            embedder=self.embedder,
            reranker=self.reranker,
            tracer=self._tracer,
        )
        self.agent, self.agent_model, client = build_agent(
            retriever=self.retriever, mode=self._mode, tracer=self._tracer
        )
        # The stub agent calls no provider, so there is nothing for readiness to reach.
        self.providers: tuple[LLMClient, ...] = (
            client.clients
            if self._mode == "live" and isinstance(client, MultiProviderRoutingClient)
            else ()
        )

    async def preload_corpus(self) -> int:
        """Ingest ``corpus_dir`` until every file is in; the number of files that failed.

        After a failure the next call ingests the whole corpus again (the upsert is
        idempotent), so a corpus that keeps failing keeps the process not ready
        instead of serving without it."""
        async with self._lock:
            if self._preloaded:
                return 0
            if not self._corpus_dir.exists():
                self._preloaded = True
                return 0
            result = await ingest_corpus(
                corpus_dir=self._corpus_dir,
                store=self.store,
                embedder=self.embedder,
            )
            self._preloaded = not result.failures
            return len(result.failures)

    def start_preload(self) -> None:
        """Ingest the corpus and load both models in the background, unless that is
        under way or done. Called at startup, and by readiness after a failure, at most
        once per readiness cache period."""
        if self.preload_state == "ok":
            return
        if self._preload_task is not None and not self._preload_task.done():
            return
        self.preload_state = "loading"
        self._preload_task = asyncio.get_running_loop().create_task(self._preload())

    async def _preload(self) -> None:
        started = time.perf_counter()
        try:
            failed_files = await self.preload_corpus()
            if failed_files:
                self.preload_state = "failed"
                _log.error("preload_failed", failed_files=failed_files)
                return
            await self.retriever.retrieve(_PRELOAD_QUERY, k=1)
        # Whatever stops the preload (a model missing from the cache, a store that is
        # not up yet) leaves the process not ready; the cause is logged, and readiness
        # starts another attempt.
        except Exception as exc:
            self.preload_state = "failed"
            _log.error("preload_failed", error_type=type(exc).__name__, error=str(exc))
            return
        self.preload_state = "ok"
        _log.info("preload_done", elapsed_ms=round((time.perf_counter() - started) * 1000, 1))

    async def ingest_bytes(
        self, *, filename: str, content: bytes, extension: str, max_documents: int
    ) -> IngestedSource:
        """Ingest a client-uploaded file.

        The URI is a **pure content address** — ``upload://<sha256><ext>`` —
        with the client-supplied filename NOT part of the URI. Same bytes
        under a different name therefore always resolve to the same
        ``document_id`` (``uuid5`` of ``identity_for(uri)``); re-uploads are
        idempotent by construction. The filename is preserved separately as
        the ``title`` for display, and the extension (parsed off the sanitised
        name) is what drives the ``SourceType`` dispatch; the caller has already
        checked it against the allow-list.

        A new document beyond ``max_documents`` raises :class:`CorpusFullError`; a
        re-upload of bytes already in the corpus does not count.
        """
        safe_name = _sanitise_filename(filename)
        digest = hashlib.sha256(content).hexdigest()
        access_uri = f"{_UPLOAD_URI_PREFIX}{digest}{extension}"
        # Check and write under one lock: both await I/O, so concurrent uploads would
        # otherwise all pass the check before any of them is stored. One process per
        # tenant (SECURITY.md) makes an in-process lock sufficient.
        async with self._ingest_lock:
            held = {d.document_id for d in await self.store.list_documents()}
            if str(document_id_for_uri(access_uri)) not in held and len(held) >= max_documents:
                raise CorpusFullError(max_documents)
            result = await ingest_uri(
                uri=access_uri,
                store=self.store,
                embedder=self.embedder,
                raw=content,
            )
        assert result.documents, "ingest_uri produced no document"
        doc = result.documents[0]
        return IngestedSource(
            uri=doc.uri,
            title=safe_name,
            document_id=str(doc.id),
            n_chunks=len(result.chunks),
        )

    async def ask(self, question: str) -> Answer:
        return await self.agent.answer(question)

    async def sources(self) -> Sequence[DocumentSummary]:
        return await self.store.list_documents()

    def provenance(self) -> Provenance:
        """Fallback provenance derived from the boot-time configuration.

        Used only by ``/status`` (and by any call path that hasn't produced
        an :class:`Answer` yet). Never a substitute for
        :meth:`provenance_for` — call sites that HAVE an Answer must use that
        one so a live-mode routing fallback is reported per request.
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
        still carries the routing decision.

        When ``provider_used`` is genuinely ``None`` — the caller built an
        Answer manually, no routing decision available — we surface
        ``agent_model="unknown"`` explicitly. We deliberately do **not**
        silently fall back to the boot-time :meth:`provenance` here: a live
        deployment with a stub-side fallback that misreports as "live" is
        exactly the class of wrong label this prevents.
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


def default_agent_mode() -> AgentMode:
    """Stub by default when no LLM key is configured — matches the CLI guard."""
    settings = get_settings()
    if settings.has_llm_api_key:
        return "live"
    return "stub"


__all__ = ["AppRuntime", "CorpusFullError", "IngestedSource", "default_agent_mode"]
