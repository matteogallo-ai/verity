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
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator

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


REFUSAL_TEXT = "I don't know based on the provided documents."
NO_CITATION_RATIONALE = "The draft answer carried no verifiable citation."


class RefusalReason(StrEnum):
    """Why an answer was refused. Distinct from :class:`ErrorKind`: a refusal is a
    decision about the evidence, an error means the system could not decide."""

    MODEL_REFUSED = "model_refused"
    BELOW_THRESHOLD = "below_threshold"
    SCORER_UNPARSEABLE = "scorer_unparseable"
    NO_CITATION = "no_citation"
    LOW_CITATION_COVERAGE = "low_citation_coverage"
    IRRELEVANT_CITATION = "irrelevant_citation"
    SUSPECTED_INJECTION = "suspected_injection"


class Confidence(BaseModel, frozen=True):
    """The scorer's confidence in the answer and the refusal decision.

    ``score`` is the scorer model's self-reported confidence in [0, 1]; it is not
    calibrated. ``refused`` is True when the scorer asked to refuse or ``score`` is
    below the configured threshold; the agent then returns an "I don't know" answer
    instead of the draft. ``rationale`` explains *why* — weak retrieval, contradictory
    evidence, question out of corpus scope.
    """

    score: float = Field(ge=0.0, le=1.0)
    refused: bool
    rationale: str
    reason: RefusalReason | None = None


class UsageStats(BaseModel, frozen=True):
    """Token and cost accounting for a single answer, aggregated across LLM calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    # Attempts the provider SDK retried before the call returned: their time is in
    # ``latency_ms``, and only this count tells a retried call from a slow one.
    retries: int = 0


class ErrorKind(StrEnum):
    """An infrastructure failure that prevented an answer. Never a refusal."""

    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_AUTH = "provider_auth"
    PROVIDER_ERROR = "provider_error"
    RETRIEVAL_UNAVAILABLE = "retrieval_unavailable"
    # A spending cap stopped the answer before its next LLM call.
    BUDGET_EXCEEDED = "budget_exceeded"


class AnswerError(BaseModel, frozen=True):
    """What failed. ``detail`` is for logs and operators, never for API clients."""

    kind: ErrorKind
    detail: str


class Answer(BaseModel, frozen=True):
    """The complete result of a query: text, citations, confidence,
    the evidence actually used, and full accounting — everything the UI and the
    eval harness need, nothing they have to reconstruct.

    ``provider_used`` and ``model_used`` are stamped by :class:`RagAgent` from
    the *synthesizer's* :class:`Completion` — the LLM call that actually produced
    the answer text. This is what makes the UI's provenance badge reflect what
    served the request, not the boot-time default. When live routing falls back
    (e.g. Anthropic unavailable → OpenAI), these fields reflect the fallback,
    not the primary. Optional (``None``) so code that builds an ``Answer``
    without a routed synthesis step (tests, persisted runs) stays valid.
    """

    query: str
    text: str
    citations: tuple[Citation, ...] = ()
    confidence: Confidence | None = None
    error: AnswerError | None = None
    hits_used: tuple[RetrievalHit, ...] = ()
    usage: UsageStats = Field(default_factory=UsageStats)
    trace_id: TraceId = Field(default_factory=new_id)
    provider_used: Provider | None = None
    model_used: str | None = None
    # Wall-clock of each agent stage that ran, in ms, so a slow answer can be explained
    # from the eval artefact instead of from logs that are not kept.
    stage_ms: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _uncited_answers_are_refusals(cls, data: Any) -> Any:
        """Exactly one of ``confidence`` / ``error``; an answer that is not refused
        and carries no citation is converted into a refusal (``no_citation``).

        The agent already refuses such drafts; this holds the invariant for every
        other way an Answer is built, so no caller can ship uncited text.
        """
        if not isinstance(data, dict):
            return data
        confidence = data.get("confidence")
        error = data.get("error")
        if (confidence is None) == (error is None):
            raise ValueError("an Answer carries exactly one of confidence or error")
        if error is not None:
            return data
        conf = (
            confidence
            if isinstance(confidence, Confidence)
            else Confidence.model_validate(confidence)
        )
        if conf.refused or data.get("citations"):
            return data
        return {
            **data,
            "text": REFUSAL_TEXT,
            "citations": (),
            "confidence": Confidence(
                score=conf.score,
                refused=True,
                rationale=NO_CITATION_RATIONALE,
                reason=RefusalReason.NO_CITATION,
            ),
        }


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
    # Examples where a provider or the store failed. They are excluded from every
    # quality metric (an outage is neither a correct nor a wrong refusal) and listed
    # here so a run with failures cannot pass for a clean one.
    errored_example_ids: tuple[str, ...] = ()


class ClaimVerdict(BaseModel, frozen=True):
    """One claim from a judged answer, with the judge's verdict on it.

    Persisted per example in :attr:`PerExampleAudit.judge_claims` so a scorecard
    reader can trace *why* a per-example ``AnswerMetrics.faithfulness`` was
    below 1.0 without re-running the judge. ``supported`` is grounding in the
    CONTEXT; ``citation_ok`` is whether the attached citation actually supports
    the claim.
    """

    claim: str
    supported: bool
    citation_ok: bool


class PerExampleAudit(BaseModel, frozen=True):
    """Full per-example audit record — the layer that lets a run of record be
    RE-READ later without re-spending on the judge.

    The published headline scorecard remains aggregated; this record is the
    audit trail that stands behind every number. Persisted verbatim under
    :attr:`EvalRun.per_audit` so ``verity eval compare`` can walk from an aggregate metric back to the exact answer, citations, hits
    and per-claim judge verdict that produced it.

    Design guarantee — see ``tests/unit/test_audit_persistence.py``:

    - Non-refused examples MUST carry non-empty ``answer_text`` + at least
      the citations the synthesizer emitted.
    - Refused examples MUST carry a non-empty ``refusal_rationale``.
    - ``judge_claims`` is ``None`` on refusals (the judge short-circuits) and
      a (possibly empty) tuple otherwise.
    """

    example_id: str
    question: str
    expected_answerable: bool
    answer_text: str
    refused: bool
    refusal_rationale: str = ""
    citations: tuple[Citation, ...] = ()
    hits_used: tuple[RetrievalHit, ...] = ()
    judge_claims: tuple[ClaimVerdict, ...] | None = None
    answer_metrics: AnswerMetrics
    usage: UsageStats = Field(default_factory=UsageStats)
    stage_ms: dict[str, float] = Field(default_factory=dict)


class EvalRun(BaseModel, frozen=True):
    """A stored eval run: the scorecard plus the environment it was produced in.
    ``compare`` reads a sequence of these to surface regressions.

    The three model-provenance fields form a complete triple:

    - ``embedding_model`` — the vector encoder used for retrieval (real component
      even in ``--judge stub`` mode).
    - ``agent_model`` — the LLM the *agent* used to decompose, synthesize, and
      self-score. When set to ``"stub-agent"`` the Scorecard's refusal numbers
      (though mechanically deterministic) reflect the *stub*'s calibration, not
      the real agent's — the CLI and README carry this nuance explicitly.
    - ``judge_model`` — the faithfulness/citation LLM judge. ``"stub-judge-v1"``
      means the answer-quality numbers are mechanical, not LLM verdicts.

    All three are persisted so a scorecard can never be interpreted without
    knowing which model produced each track.

    ``per_audit`` is the audit trail — one :class:`PerExampleAudit` per dataset
    example, capturing the raw answer text, citations, hits, and per-claim
    judge verdict. Empty by default for backwards compatibility with pre-v1.0.0
    persisted runs. See :class:`PerExampleAudit` for the auditability contract.
    """

    scorecard: Scorecard
    embedding_model: str
    judge_model: str
    agent_model: str = "unknown"
    notes: str | None = None
    per_audit: tuple[PerExampleAudit, ...] = ()
