"""The refusal text is configurable: an association's members read it in their
language, and it tells them whom to ask instead."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from tests.unit.test_agent import _seed_retriever
from tests.unit.test_error_vs_refusal import _ANSWERABLE, _router_with_synthesis
from verity.agent.agent import RagAgent, create_default_agent
from verity.config import Settings, get_settings
from verity.types import REFUSAL_TEXT

_FR = (
    "Les documents fournis ne permettent pas de répondre. "
    "Contactez le secrétariat de l'association."
)
_UNCITED = json.dumps({"answer": "Thirty days.", "citations": []})


async def _refusal(agent: RagAgent) -> str:
    answer = await agent.answer(_ANSWERABLE)
    assert answer.confidence is not None and answer.confidence.refused
    return answer.text


@pytest.mark.asyncio
async def test_the_default_refusal_text_is_unchanged() -> None:
    router, _ = _router_with_synthesis(_UNCITED)
    agent = create_default_agent(retriever=await _seed_retriever(), client=router)
    assert await _refusal(agent) == REFUSAL_TEXT


@pytest.mark.asyncio
async def test_the_setting_sets_the_refusal_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VERITY_REFUSAL_TEXT", _FR)
    get_settings.cache_clear()
    router, _ = _router_with_synthesis(_UNCITED)
    agent = create_default_agent(retriever=await _seed_retriever(), client=router)
    assert await _refusal(agent) == _FR


@pytest.mark.asyncio
async def test_an_agent_can_carry_its_own_refusal_text() -> None:
    """One process can serve several corpora, each with its own contact."""
    router, _ = _router_with_synthesis(_UNCITED)
    default = create_default_agent(retriever=await _seed_retriever(), client=router)
    agent = RagAgent(
        retriever=default._retriever,
        decomposer=default._decomposer,
        synthesizer=default._synthesizer,
        scorer=default._scorer,
        refusal_text=_FR,
    )
    assert await _refusal(agent) == _FR


@pytest.mark.parametrize("text", ["", "   "])
def test_a_blank_refusal_text_is_rejected(text: str) -> None:
    with pytest.raises(ValidationError):
        Settings(refusal_text=text)
