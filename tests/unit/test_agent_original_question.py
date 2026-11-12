"""VERITY_RETRIEVE_ORIGINAL_QUESTION: the question as asked is retrieved too.

A decomposition can drop the term that finds the passage: on the landscapers' CCT the
question on Sunday work finds article 9.1 at rank 4 by itself, and its sub-questions
did not find it at all. With the setting, retrieval runs on the question and on each
sub-question; hits are already deduplicated by chunk.
"""

from __future__ import annotations

import pytest

from verity.agent.agent import create_default_agent
from verity.agent.scripted_stub import build_stub_router
from verity.types import RetrievalHit


class _Decomposer:
    last_usage = None

    def __init__(self, sub_questions: list[str]) -> None:
        self._sub_questions = sub_questions

    async def decompose(self, question: str) -> list[str]:
        return list(self._sub_questions)


class _Retriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def retrieve(self, query: str, *, k: int) -> list[RetrievalHit]:
        self.queries.append(query)
        return []


async def _queries(question: str, sub_questions: list[str]) -> list[str]:
    retriever = _Retriever()
    agent = create_default_agent(retriever=retriever, client=build_stub_router())  # type: ignore[arg-type]
    agent._decomposer = _Decomposer(sub_questions)  # type: ignore[assignment]
    await agent.answer(question)
    return retriever.queries


@pytest.mark.asyncio
async def test_by_default_only_the_sub_questions_are_retrieved() -> None:
    assert await _queries("Q?", ["a?", "b?"]) == ["a?", "b?"]


@pytest.mark.asyncio
async def test_the_setting_retrieves_the_question_first_then_its_parts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from verity.config import get_settings

    monkeypatch.setenv("VERITY_RETRIEVE_ORIGINAL_QUESTION", "true")
    get_settings.cache_clear()
    assert await _queries("Q?", ["a?", "b?"]) == ["Q?", "a?", "b?"]


@pytest.mark.asyncio
async def test_a_question_kept_whole_is_retrieved_once(monkeypatch: pytest.MonkeyPatch) -> None:
    from verity.config import get_settings

    monkeypatch.setenv("VERITY_RETRIEVE_ORIGINAL_QUESTION", "true")
    get_settings.cache_clear()
    assert await _queries("Q?", ["Q?"]) == ["Q?"]
    assert await _queries("Q?", ["a?", "a?"]) == ["Q?", "a?"]
