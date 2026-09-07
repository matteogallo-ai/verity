"""HTTP response models — thin JSON mirrors of the domain types.

Rule of iron: no computed field, no fabricated score, no invented composite
"confidence". Every field on the wire has a direct counterpart in
:mod:`verity.types`. The only additions are:

- ``RetrievedChunkView.cited`` — a boolean derived by structural set-membership
  (``chunk_id in {c.chunk_id for c in citations}``). Marked in the schema as
  such so the UI can render "cited / not-cited" without recomputing it.
- ``Provenance`` — a small dataclass carrying the (agent_model, is_stub,
  embedding_model, reranker_model) triple as observed for THIS specific
  request. It reflects what actually ran, not a static Settings default.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from verity.types import Answer, RetrievalHit


class Provenance(BaseModel, frozen=True):
    """What actually served this request.

    ``agent_model`` and ``is_stub`` are derived from :attr:`Answer.provider_used`
    / :attr:`Answer.model_used` when the answer carries them — i.e. from the
    concrete :class:`Completion` returned by whichever ``LLMClient`` the router
    used for the synthesis step. In live mode this correctly reflects a
    ``ProviderUnavailableError`` fallback (e.g. Anthropic → OpenAI). When an
    Answer is built by a code path that doesn't stamp a provider (older tests,
    ``/status`` before any request), the fields fall back to the boot-time
    :class:`AppRuntime` configuration.
    """

    agent_model: str
    is_stub: bool
    embedding_model: str
    reranker_model: str


class CitationView(BaseModel, frozen=True):
    chunk_id: str
    document_id: str
    quote: str
    char_start: int
    char_end: int


class ConfidenceView(BaseModel, frozen=True):
    """Verbatim from :class:`verity.types.Confidence`. ``score`` is the calibrated
    confidence produced by the scorer for this request (real value, not fabricated).
    ``refused`` is the load-bearing headline behaviour: True → the agent honoured
    its refusal threshold rather than fabricate."""

    score: float = Field(ge=0.0, le=1.0)
    refused: bool
    rationale: str


class UsageStatsView(BaseModel, frozen=True):
    input_tokens: int
    output_tokens: int
    llm_calls: int
    cost_usd: float
    latency_ms: float


class RetrievedChunkView(BaseModel, frozen=True):
    """A chunk in the evidence set the agent reasoned over — with its rerank score
    and a structurally-derived ``cited`` flag (``True`` iff at least one citation
    points at this chunk)."""

    chunk_id: str
    document_id: str
    text: str
    section: str | None
    ordinal: int
    char_start: int
    char_end: int
    score: float
    rank: int
    kind: str  # "dense" | "sparse" | "fused" | "reranked"
    cited: bool


class AnswerView(BaseModel, frozen=True):
    query: str
    text: str
    citations: tuple[CitationView, ...]
    confidence: ConfidenceView
    hits_used: tuple[RetrievedChunkView, ...]
    usage: UsageStatsView
    trace_id: str
    # Stamped by RagAgent from the synthesizer's Completion — the LLM call
    # that actually produced ``text``. In stub mode this reads "local" /
    # "stub". In live mode with a router fallback, this reflects the fallback.
    provider_used: str | None = None
    model_used: str | None = None


class AskResponse(BaseModel, frozen=True):
    answer: AnswerView
    provenance: Provenance


class CorpusSummary(BaseModel, frozen=True):
    n_documents: int
    n_chunks: int
    sources: tuple[str, ...]  # ordered list of ingested URIs (paths + upload://…)


class IngestResponse(BaseModel, frozen=True):
    document_id: str
    uri: str
    title: str | None
    n_chunks: int
    corpus: CorpusSummary


class StatusResponse(BaseModel, frozen=True):
    version: str
    provenance: Provenance
    corpus: CorpusSummary
    otel_endpoint: str | None


# --------------------------------------------------------------------------------------
# Conversion helpers
# --------------------------------------------------------------------------------------


def answer_to_view(answer: Answer) -> AnswerView:
    """Serialise an :class:`Answer` into ``AnswerView``, deriving ``cited`` per chunk.

    The derivation is purely structural: a chunk is marked cited iff its id
    appears in at least one citation. No score is invented; every numeric on
    the wire is copied verbatim.
    """
    cited_ids = {str(c.chunk_id) for c in answer.citations}
    hits_view = tuple(_hit_to_view(hit, cited_ids) for hit in answer.hits_used)
    citations_view = tuple(
        CitationView(
            chunk_id=str(c.chunk_id),
            document_id=str(c.document_id),
            quote=c.quote,
            char_start=c.char_start,
            char_end=c.char_end,
        )
        for c in answer.citations
    )
    return AnswerView(
        query=answer.query,
        text=answer.text,
        citations=citations_view,
        confidence=ConfidenceView(
            score=answer.confidence.score,
            refused=answer.confidence.refused,
            rationale=answer.confidence.rationale,
        ),
        hits_used=hits_view,
        usage=UsageStatsView(
            input_tokens=answer.usage.input_tokens,
            output_tokens=answer.usage.output_tokens,
            llm_calls=answer.usage.llm_calls,
            cost_usd=answer.usage.cost_usd,
            latency_ms=answer.usage.latency_ms,
        ),
        provider_used=(answer.provider_used.value if answer.provider_used is not None else None),
        model_used=answer.model_used,
        trace_id=str(answer.trace_id),
    )


def _hit_to_view(hit: RetrievalHit, cited_ids: set[str]) -> RetrievedChunkView:
    return RetrievedChunkView(
        chunk_id=str(hit.chunk.id),
        document_id=str(hit.chunk.document_id),
        text=hit.chunk.text,
        section=hit.chunk.section,
        ordinal=hit.chunk.ordinal,
        char_start=hit.chunk.char_start,
        char_end=hit.chunk.char_end,
        score=hit.score,
        rank=hit.rank,
        kind=hit.kind.value,
        cited=str(hit.chunk.id) in cited_ids,
    )


def _model_dump(m: BaseModel) -> dict[str, Any]:
    """Shim to avoid mypy noise across pydantic versions."""
    return m.model_dump()


__all__ = [
    "AnswerView",
    "AskResponse",
    "CitationView",
    "ConfidenceView",
    "CorpusSummary",
    "IngestResponse",
    "Provenance",
    "RetrievedChunkView",
    "StatusResponse",
    "UsageStatsView",
    "answer_to_view",
]
