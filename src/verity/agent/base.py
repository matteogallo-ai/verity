"""Agentic RAG contracts.

The agent is not single-shot retrieval. It decomposes a complex question into
sub-questions, retrieves per sub-question, combines evidence, drafts a cited answer,
then scores its own confidence and **refuses** rather than fabricating when the evidence
is weak. Each of those is a named, individually testable protocol so the eval harness
can attribute a regression to decomposition, synthesis, or calibration.

The prompts driving decomposition, the faithfulness judge, and confidence scoring are
authored in PromptLang (see ``prompts/``) and consumed through the runtime — Verity
dogfoods PromptLang rather than embedding f-strings.

Implementations land in S3.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from verity.types import Answer, Confidence, RetrievalHit


@runtime_checkable
class QuestionDecomposer(Protocol):
    """Splits a complex question into ordered sub-questions. Returns a single-element
    list for simple questions (no forced multi-step)."""

    async def decompose(self, question: str) -> list[str]: ...


@runtime_checkable
class ConfidenceScorer(Protocol):
    """Produces a calibrated :class:`Confidence` from the question, drafted answer, and
    the evidence actually used. Owns the refusal decision against the configured
    threshold."""

    async def score(
        self, question: str, draft_answer: str, hits: list[RetrievalHit]
    ) -> Confidence: ...


@runtime_checkable
class Agent(Protocol):
    """Top-level orchestrator: question in, fully-formed :class:`Answer` out.

    Guarantees on the returned Answer:
      * every non-refusal answer carries ≥1 citation mapped to an exact passage;
      * ``confidence.refused`` is True whenever confidence < threshold, and the text is
        an honest "I don't know" rather than a hallucinated answer;
      * ``usage`` and ``trace_id`` are always populated for observability and eval.
    """

    async def answer(self, question: str) -> Answer: ...
