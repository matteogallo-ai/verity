"""QuestionDecomposer: JSON parsing + degrade to [question] on malformed output."""

from __future__ import annotations

import pytest

from verity.agent.decomposer import LLMQuestionDecomposer
from verity.llm.clients import StubLLMClient
from verity.llm.routing import MultiProviderRoutingClient


def _router(responder) -> MultiProviderRoutingClient:  # type: ignore[no-untyped-def]
    return MultiProviderRoutingClient([StubLLMClient(responder)])


@pytest.mark.asyncio
async def test_decomposer_returns_json_list() -> None:
    router = _router(lambda _: '["sub 1", "sub 2"]')
    decomposer = LLMQuestionDecomposer(router)
    assert await decomposer.decompose("q?") == ["sub 1", "sub 2"]


@pytest.mark.asyncio
async def test_decomposer_extracts_json_from_prose() -> None:
    router = _router(lambda _: 'Sure! Here it is: ["only one"] hope this helps.')
    decomposer = LLMQuestionDecomposer(router)
    assert await decomposer.decompose("q?") == ["only one"]


@pytest.mark.asyncio
async def test_decomposer_degrades_to_original_on_malformed() -> None:
    router = _router(lambda _: "not JSON at all")
    decomposer = LLMQuestionDecomposer(router)
    assert await decomposer.decompose("original?") == ["original?"]


@pytest.mark.asyncio
async def test_decomposer_caps_at_max_steps() -> None:
    router = _router(lambda _: '["a", "b", "c", "d", "e", "f", "g"]')
    decomposer = LLMQuestionDecomposer(router, max_steps=3)
    assert await decomposer.decompose("q?") == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_decomposer_ignores_non_string_elements() -> None:
    router = _router(lambda _: '["a", 42, null, "b"]')
    decomposer = LLMQuestionDecomposer(router)
    # A mixed list fails validation → degrades to [question].
    assert await decomposer.decompose("original?") == ["original?"]
