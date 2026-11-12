"""HTTP endpoints for the demo UI — thin serializers over :class:`AppRuntime`.

Every handler here is a wrapper: it reads a request, calls the shared runtime,
and dumps the domain object through :mod:`verity.api.models`. There is no
business logic in this module — enforcing "the CLI and the API cannot diverge"
is exactly what makes the app safe to demo.
"""

from __future__ import annotations

from typing import Annotated, cast

import structlog
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from verity import __version__
from verity.api.guards import (
    SpendBudget,
    UploadRejectedError,
    check_upload,
    require_api_key,
    retry_after_header,
)
from verity.api.models import (
    AskResponse,
    CorpusSummary,
    IngestResponse,
    StatusResponse,
    answer_to_view,
)
from verity.api.state import AppRuntime, CorpusFullError
from verity.config import Settings
from verity.retrieval.base import RetrievalUnavailableError, StoreSchemaError
from verity.types import ErrorKind

router = APIRouter()
_log = structlog.get_logger("verity.api")


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


async def _corpus_summary(runtime: AppRuntime) -> CorpusSummary:
    sources = await runtime.sources()
    return CorpusSummary(
        n_documents=len(sources),
        n_chunks=sum(s.n_chunks for s in sources),
        sources=tuple(s.uri for s in sources),
    )


@router.get("/status", response_model=StatusResponse, dependencies=[Depends(require_api_key)])
async def status_endpoint(request: Request) -> StatusResponse:
    """Status: real corpus counts, real provenance triple, real
    otel endpoint config. No hard-coded ``"ok"``.

    Behind the key because it lists every document URI and the models in use;
    ``/health`` and ``/ready`` are the public probes."""
    runtime = _runtime(request)
    settings: Settings = request.app.state.settings
    # First hit doubles as a lazy preload — subsequent hits are free.
    await runtime.preload_corpus()
    return StatusResponse(
        version=__version__,
        provenance=runtime.provenance(),
        corpus=await _corpus_summary(runtime),
        otel_endpoint=settings.otel_endpoint,
        language=settings.language,
        suggested_questions=runtime.suggested_questions(),
        ui_mode=settings.ui_mode,
        corpus_name=settings.corpus_name,
    )


@router.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
async def ingest_endpoint(
    request: Request,
    file: Annotated[UploadFile, File(description="Document to ingest (pdf, docx, md, txt).")],
) -> IngestResponse:
    """Ingest one uploaded file into the configured store.

    The document id is ``uuid5(NAMESPACE_URL, "upload:/<sha256><ext>")``: the same
    bytes with the same extension always yield the same id, whatever the file name
    or machine. The extension is kept because it selects the parser.

    Limits (SECURITY.md): size (413, enforced on the request body before parsing and
    again here), type allow-list checked on extension, declared MIME and content
    (415), corpus size (409). A parse failure is a generic 422: the parser's message
    names temporary files and library internals, so it goes to the log only.
    """
    runtime = _runtime(request)
    settings: Settings = request.app.state.settings
    await runtime.preload_corpus()

    content = await file.read(settings.api_max_upload_bytes + 1)
    if not content:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(content) > settings.api_max_upload_bytes:
        raise HTTPException(status_code=413, detail="Upload too large.")
    filename = file.filename or "upload"
    try:
        extension = check_upload(filename, file.content_type, content)
    except UploadRejectedError as exc:
        raise HTTPException(status_code=415, detail=exc.message) from exc
    try:
        record = await runtime.ingest_bytes(
            filename=filename,
            content=content,
            extension=extension,
            max_documents=settings.api_max_documents,
        )
    except CorpusFullError as exc:
        raise HTTPException(status_code=409, detail="The corpus is full.") from exc
    except (RetrievalUnavailableError, StoreSchemaError):
        # The store failed, not the document: the app's 503 handler answers.
        raise
    except Exception as exc:
        # Parsers (docling, python-docx, decoders) raise many unrelated types; any of
        # them means "this document could not be read". Logged with the traceback,
        # re-raised as a typed 422 without the message.
        _log.warning("ingest_failed", filename=filename, exc_info=exc)
        raise HTTPException(status_code=422, detail="The document could not be parsed.") from exc

    return IngestResponse(
        document_id=record.document_id,
        uri=record.uri,
        title=record.title,
        n_chunks=record.n_chunks,
        corpus=await _corpus_summary(runtime),
    )


@router.post("/ask", response_model=AskResponse)
async def ask_endpoint(
    request: Request,
    body: AskRequest,
    principal: Annotated[str, Depends(require_api_key)],
) -> AskResponse:
    """One agent turn. Returns the full :class:`AnswerView` + the
    :class:`Provenance` **derived per-request** from the synthesizer's actual
    :class:`Completion` via :meth:`AppRuntime.provenance_for`. Provenance is
    never separated from the answer on the wire so the UI can't lift a
    number without its badge, and it correctly reflects a routing fallback
    in live mode (Anthropic → OpenAI) rather than a static boot-time value.
    """
    runtime = _runtime(request)
    limit = cast(Settings, request.app.state.settings).api_max_question_chars
    if len(body.question) > limit:
        raise HTTPException(status_code=422, detail=f"question exceeds {limit} characters.")
    budget: SpendBudget = request.app.state.spend_budget
    retry_after = budget.exhausted(principal)
    if retry_after is not None:
        # 429, not 402: the budget refills as the window slides, which is what
        # Retry-After expresses; 402 has no standard semantics clients act on.
        raise HTTPException(
            status_code=429,
            detail="Spending budget exceeded.",
            headers=retry_after_header(retry_after),
        )
    await runtime.preload_corpus()
    answer = await runtime.ask(body.question)
    # Before the error check: a failed answer may have paid for its first calls.
    budget.record(principal, answer.usage.cost_usd)
    if answer.error is not None and answer.error.kind is ErrorKind.INTERNAL:
        # A defect is the server's fault, not an outage to retry: the generic 500.
        raise HTTPException(
            status_code=500,
            detail={"error": answer.error.kind.value, "trace_id": str(answer.trace_id)},
        )
    if answer.error is not None:
        # An outage is a 503, never a refusal and never a 500. The detail stays in
        # the logs (bound to trace_id by the agent); the client gets the id only.
        raise HTTPException(
            status_code=503,
            detail={
                "error": answer.error.kind.value,
                "message": "The answer could not be produced; retry later.",
                "trace_id": str(answer.trace_id),
            },
        )
    return AskResponse(
        answer=answer_to_view(answer),
        provenance=runtime.provenance_for(answer),
    )


__all__ = ["router"]
