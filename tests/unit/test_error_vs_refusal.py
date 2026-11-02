"""An uncited draft is always a refusal (reason ``no_citation``), never an answer."""

from __future__ import annotations

import json

import pytest

from tests.unit.test_agent import _seed_retriever
from verity.agent.agent import create_default_agent
from verity.agent.scorer import LLMConfidenceScorer
from verity.agent.scripted_stub import scripted_responder
from verity.llm.clients import StubLLMClient
from verity.llm.routing import MultiProviderRoutingClient
from verity.types import (
    NO_CITATION_RATIONALE,
    REFUSAL_TEXT,
    Answer,
    Confidence,
    Message,
    RefusalReason,
)

_ANSWERABLE = "What is the notice period for terminating the master services agreement?"


# --- Answer invariants ----------------------------------------------------------------


def test_uncited_unrefused_answer_becomes_a_no_citation_refusal() -> None:
    answer = Answer(
        query="q",
        text="A fluent answer with nothing behind it.",
        confidence=Confidence(score=0.97, refused=False, rationale="looks fine"),
    )
    assert answer.confidence is not None
    assert answer.confidence.refused
    assert answer.confidence.reason is RefusalReason.NO_CITATION
    assert answer.confidence.rationale == NO_CITATION_RATIONALE
    assert answer.text == REFUSAL_TEXT


# --- Agent: no_citation gate ----------------------------------------------------------


def _router_with_synthesis(synthesis_text: str) -> tuple[MultiProviderRoutingClient, list[str]]:
    """Scripted stub for every role except synthesis, which returns ``synthesis_text``.
    Records which roles were called."""
    scripted = scripted_responder()
    roles: list[str] = []

    def _respond(messages: list[Message]) -> str:
        system = messages[0].content.lower()
        if "draft concise" in system:
            roles.append("synthesize")
            return synthesis_text
        roles.append("confidence" if "assess whether" in system else "decompose")
        return scripted(messages)

    return MultiProviderRoutingClient([StubLLMClient(_respond)]), roles


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "synthesis_text",
    [
        "Sure! The notice period is thirty days.",  # not JSON: never shipped
        json.dumps({"answer": "Thirty days.", "citations": []}),  # JSON, no citation
        json.dumps(  # citation whose quote is not in the chunk: dropped
            {"answer": "Thirty days.", "citations": [{"quote": "ninety", "chunk_index": 0}]}
        ),
    ],
    ids=["unparseable", "no-citations", "invented-quote"],
)
async def test_agent_refuses_an_uncited_draft_without_asking_the_scorer(
    synthesis_text: str,
) -> None:
    router, roles = _router_with_synthesis(synthesis_text)
    agent = create_default_agent(retriever=await _seed_retriever(), client=router)

    answer = await agent.answer(_ANSWERABLE)

    assert answer.confidence is not None
    assert answer.confidence.refused
    assert answer.confidence.reason is RefusalReason.NO_CITATION
    assert answer.text == REFUSAL_TEXT
    assert "confidence" not in roles, "an uncited draft must not reach the scorer"


@pytest.mark.asyncio
async def test_scorer_reports_why_it_refused() -> None:
    def _scorer(payload: object) -> LLMConfidenceScorer:
        text = payload if isinstance(payload, str) else json.dumps(payload)
        return LLMConfidenceScorer(
            MultiProviderRoutingClient([StubLLMClient(lambda _: text)]), threshold=0.55
        )

    below = await _scorer({"score": 0.2, "refused": False, "rationale": "thin"}).score("q", "a", [])
    model = await _scorer({"score": 0.9, "refused": True, "rationale": "off"}).score("q", "a", [])
    broken = await _scorer("no json").score("q", "a", [])
    fine = await _scorer({"score": 0.9, "refused": False, "rationale": "ok"}).score("q", "a", [])
    assert below.reason is RefusalReason.BELOW_THRESHOLD
    assert model.reason is RefusalReason.MODEL_REFUSED
    assert broken.reason is RefusalReason.SCORER_UNPARSEABLE
    assert fine.reason is None and not fine.refused
