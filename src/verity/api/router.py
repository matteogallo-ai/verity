"""HTTP endpoints for the demo UI — thin serializers over :class:`AppRuntime`.

Every handler here is a wrapper: it reads a request, calls the shared runtime,
and dumps the domain object through :mod:`verity.api.models`. There is no
business logic in this module — enforcing "the CLI and the API cannot diverge"
is exactly what makes the app safe to demo.
"""

from __future__ import annotations

from typing import Annotated, cast

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from verity import __version__
from verity.api.models import (
    AskResponse,
    CorpusSummary,
    IngestResponse,
    StatusResponse,
    answer_to_view,
)
from verity.api.state import AppRuntime
from verity.config import get_settings

router = APIRouter()


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


def _runtime(request: Request) -> AppRuntime:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(
            status_code=503,
            detail="Runtime not initialised (server started without --api mode?)",
        )
    return cast(AppRuntime, runtime)


def _corpus_summary(runtime: AppRuntime) -> CorpusSummary:
    sources = runtime.sources()
    return CorpusSummary(
        n_documents=len(sources),
        n_chunks=sum(s.n_chunks for s in sources),
        sources=tuple(s.uri for s in sources),
    )


@router.get("/status", response_model=StatusResponse)
async def status_endpoint(request: Request) -> StatusResponse:
    """Honest status: real corpus counts, real provenance triple, real
    otel endpoint config. No hard-coded ``"ok"``."""
    runtime = _runtime(request)
    settings = get_settings()
    # First hit doubles as a lazy preload — subsequent hits are free.
    await runtime.preload_corpus()
    return StatusResponse(
        version=__version__,
        provenance=runtime.provenance(),
        corpus=_corpus_summary(runtime),
        otel_endpoint=settings.otel_endpoint,
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(
    request: Request,
    file: Annotated[UploadFile, File(description="Document to ingest (pdf, docx, md, txt).")],
) -> IngestResponse:
    """Ingest one uploaded file into the in-memory store.

    The document id is derived from ``uuid5(NAMESPACE_URL, "upload://<sha256>/<name>")``
    so the same bytes always yield the same id, regardless of the machine or
    original filename path.
    """
    runtime = _runtime(request)
    await runtime.preload_corpus()

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty upload.")
    try:
        record = await runtime.ingest_bytes(filename=file.filename or "upload", content=content)
    except Exception as exc:  # translate any parse error into a 4xx for the UI
        raise HTTPException(status_code=422, detail=f"Ingestion failed: {exc}") from exc

    return IngestResponse(
        document_id=record.document_id,
        uri=record.uri,
        title=record.title,
        n_chunks=record.n_chunks,
        corpus=_corpus_summary(runtime),
    )


@router.post("/ask", response_model=AskResponse)
async def ask_endpoint(request: Request, body: AskRequest) -> AskResponse:
    """One agent turn. Returns the full :class:`AnswerView` + the
    :class:`Provenance` **derived per-request** from the synthesizer's actual
    :class:`Completion` via :meth:`AppRuntime.provenance_for`. Provenance is
    never separated from the answer on the wire so the UI can't lift a
    chiffre without its badge, and it correctly reflects a routing fallback
    in live mode (Anthropic → OpenAI) rather than a static boot-time value.
    """
    runtime = _runtime(request)
    await runtime.preload_corpus()
    answer = await runtime.ask(body.question)
    return AskResponse(
        answer=answer_to_view(answer),
        provenance=runtime.provenance_for(answer),
    )


__all__ = ["router"]
