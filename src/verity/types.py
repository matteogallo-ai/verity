"""Core domain model for Verity.

These types are the contracts between every stage of the pipeline. Implementations
live in the stage packages (``ingestion``, ``retrieval``, ``agent``, ``eval`` …) but
they all speak in terms of the frozen models defined here. Keeping the domain model
in one place, immutable, and provider-agnostic is deliberate: it is what lets a
retriever, a reranker, an agent, and the eval harness compose without leaking each
other's implementation details.

Nothing in this module imports a provider SDK, a database driver, or a web framework.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------------------
# Identifiers
# --------------------------------------------------------------------------------------

DocumentId = UUID
ChunkId = UUID
TraceId = UUID


def new_id() -> UUID:
    return uuid4()


def _utcnow() -> datetime:
    return datetime.now(UTC)


# --------------------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------------------


class SourceType(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MARKDOWN = "markdown"
    WEB = "web"


class Document(BaseModel, frozen=True):
    """A source document after parsing, before chunking."""

    id: DocumentId = Field(default_factory=new_id)
    source_type: SourceType
    uri: str  # file path or URL — provenance, always preserved for citations
    title: str | None = None
    text: str
    metadata: dict[str, str] = Field(default_factory=dict)
    ingested_at: datetime = Field(default_factory=_utcnow)


class Chunk(BaseModel, frozen=True):
    """A retrievable unit produced by structure-aware chunking.

    ``char_start``/``char_end`` are offsets into the *parent document text*. They are
    non-negotiable: they are what makes a citation point to an exact passage the UI can
    highlight, rather than to "somewhere in this document".
    """

    id: ChunkId = Field(default_factory=new_id)
    document_id: DocumentId
    text: str
    ordinal: int  # position of the chunk within its document, 0-based
    char_start: int
    char_end: int
    section: str | None = None  # e.g. heading path "1.2 Risk factors"
    page: int | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class EmbeddedChunk(BaseModel, frozen=True):
    """A chunk with its dense vector attached. Kept separate from ``Chunk`` so that
    chunking has no dependency on the embedding model."""

    chunk: Chunk
    embedding: tuple[float, ...]
    model: str  # embedding model id, stored for reproducibility of eval runs


# --------------------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------------------


class RetrieverKind(StrEnum):
    DENSE = "dense"
    SPARSE = "sparse"
    FUSED = "fused"  # after reciprocal rank fusion
    RERANKED = "reranked"  # after cross-encoder rerank


class RetrievalHit(BaseModel, frozen=True):
    """A retrieved chunk with the score and the stage that produced it.

    ``score`` semantics depend on ``kind`` (cosine similarity for dense, BM25 for
    sparse, fused RRF score, reranker logit). The eval harness and the observability
    layer both read ``kind`` to interpret ``score`` — never assume a single scale.
    """

    chunk: Chunk
    score: float
    kind: RetrieverKind
    rank: int  # 0-based rank within the result set that produced this hit


# --------------------------------------------------------------------------------------
# Answering
# --------------------------------------------------------------------------------------


class Citation(BaseModel, frozen=True):
    """An inline citation mapping a claim in the answer to an exact source passage."""

    chunk_id: ChunkId
    document_id: DocumentId
    quote: str  # verbatim span from the chunk supporting the claim
    char_start: int  # offset into the parent document text
    char_end: int


class Confidence(BaseModel, frozen=True):
    """Calibrated confidence in the answer.

    ``refused`` is the headline behaviour of Verity: when ``score`` falls below the
    configured threshold the agent returns an "I don't know" answer instead of a
    fabricated one. ``rationale`` explains *why* — weak retrieval, contradictory
    evidence, question out of corpus scope.
    """

    score: float = Field(ge=0.0, le=1.0)
    refused: bool
    rationale: str


class UsageStats(BaseModel, frozen=True):
    """Token and cost accounting for a single answer, aggregated across LLM calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0


class Answer(BaseModel, frozen=True):
    """The complete result of a query: text, citations, calibrated confidence,
    the evidence actually used, and full accounting — everything the UI and the
    eval harness need, nothing they have to reconstruct."""

    query: str
    text: str
    citations: tuple[Citation, ...] = ()
    confidence: Confidence
    hits_used: tuple[RetrievalHit, ...] = ()
    usage: UsageStats = Field(default_factory=UsageStats)
    trace_id: TraceId = Field(default_factory=new_id)


# --------------------------------------------------------------------------------------
# LLM runtime
# --------------------------------------------------------------------------------------


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel, frozen=True):
    role: Role
    content: str


class Provider(StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    LOCAL = "local"


class Completion(BaseModel, frozen=True):
    text: str
    provider: Provider
    model: str
    usage: UsageStats = Field(default_factory=UsageStats)


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------


class EvalExample(BaseModel, frozen=True):
    """One labeled example in the eval dataset.

    ``answerable`` is central to Verity's thesis: the dataset intentionally contains
    questions whose answer is *not* in the corpus, so the harness can measure refusal
    calibration (does the system say "I don't know" exactly when it should?), not only
    faithfulness on answerable questions.
    """

    id: str
    question: str
    answerable: bool
    reference_answer: str | None = None  # None iff not answerable
    relevant_chunk_ids: tuple[ChunkId, ...] = ()  # gold retrieval labels
    tags: tuple[str, ...] = ()


class RetrievalMetrics(BaseModel, frozen=True):
    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    k: int


class AnswerMetrics(BaseModel, frozen=True):
    faithfulness: float  # fraction of answer claims grounded in retrieved context
    citation_accuracy: float  # fraction of citations that actually support their claim
    hallucination_rate: float  # answers with ≥1 unsupported claim / answerable answers
    refusal_precision: float  # of refusals, fraction that *should* have been refused
    refusal_recall: float  # of unanswerable questions, fraction correctly refused


class LatencyMetrics(BaseModel, frozen=True):
    p50_ms: float
    p95_ms: float
    p99_ms: float


class Scorecard(BaseModel, frozen=True):
    """The single artefact the eval harness produces and the README publishes.

    Persisted alongside the git SHA so eval runs are comparable over time
    (regression tracking). This is the object that mirrors how a real AI team
    tracks quality across commits.
    """

    git_sha: str
    created_at: datetime = Field(default_factory=_utcnow)
    dataset: str
    n_examples: int
    retrieval: RetrievalMetrics
    answer: AnswerMetrics
    latency: LatencyMetrics
    cost_per_query_usd: float
    # Cold-start latency of the first agent call (model loads, tokeniser init).
    # Separated from ``latency`` so p50/p95/p99 reflect the *steady-state*
    # serving latency an operator would see in production. ``None`` when no
    # warmup was performed (measurement is combined into the percentiles then).
    cold_start_ms: float | None = None


class EvalRun(BaseModel, frozen=True):
    """A stored eval run: the scorecard plus the environment it was produced in.
    ``compare`` reads a sequence of these to surface regressions.

    The three model-provenance fields form a complete triple:

    - ``embedding_model`` — the vector encoder used for retrieval (real component
      even in ``--judge stub`` mode).
    - ``agent_model`` — the LLM the *agent* used to decompose, synthesize, and
      self-score. When set to ``"stub-agent"`` the Scorecard's refusal chiffres
      (though mechanically deterministic) reflect the *stub*'s calibration, not
      the real agent's — the CLI and README carry this nuance explicitly.
    - ``judge_model`` — the faithfulness/citation LLM judge. ``"stub-judge-v1"``
      means the answer-quality numbers are mechanical, not LLM verdicts.

    All three are persisted so a scorecard can never be interpreted without
    knowing which model produced each track.
    """

    scorecard: Scorecard
    embedding_model: str
    judge_model: str
    agent_model: str = "unknown"
    notes: str | None = None
