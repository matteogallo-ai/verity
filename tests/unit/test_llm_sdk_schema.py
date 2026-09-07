"""SDK-signature smoke tests for the 4 real LLM call sites.

Motivation — the S7 hotfix incident: ``anthropic`` 1.x removed ``temperature``
from the typed ``AsyncMessages.create`` signature. Every unit test in the
suite drives the pipeline through :class:`StubLLMClient`, so the real Anthropic
call shape (kwargs actually delivered to the SDK) was never exercised until
Matteo tried the first live run and crashed with a ``TypeError``.

This module fixes the class of bug by asserting, at import time and in every
CI run, that the concrete kwargs :class:`AnthropicClient` / :class:`OpenAIClient`
pass to ``messages.create`` / ``chat.completions.create`` are a subset of the
pinned SDK's real signature. A future dependency bump that renames or drops a
kwarg will fail this test loudly instead of at the first live spend.

Zero network, zero API key required. Each test:

1. Introspects the pinned SDK's signature via :func:`inspect.signature`.
2. Monkey-patches the client's factory to return an :class:`AsyncMock`
   ``spec``ed against that real signature.
3. Runs one call through each of the 4 LLM points (agent decompose, agent
   synthesize, agent confidence, judge) with a scripted context.
4. Asserts on the exact kwargs the mock was invoked with, then binds them to
   the real signature via ``sig.bind()`` — an unknown kwarg raises ``TypeError``
   and the test fails, exactly like the live path would.
"""

from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from verity.llm.clients import AnthropicClient, OpenAIClient
from verity.types import Message, Role


def _bind_or_die(sig: inspect.Signature, kwargs: dict[str, Any]) -> None:
    """Bind ``kwargs`` to ``sig``; raise if any are unknown or missing required."""
    try:
        sig.bind(**kwargs)
    except TypeError as exc:
        raise AssertionError(
            f"kwargs {list(kwargs)} do not match the pinned SDK signature: {exc}"
        ) from exc


def _anthropic_create_signature() -> inspect.Signature:
    """Signature of ``AsyncMessages.create`` in the pinned anthropic SDK."""
    import anthropic

    dummy = anthropic.AsyncAnthropic(api_key="test-key-not-used")
    return inspect.signature(dummy.messages.create)


def _openai_create_signature() -> inspect.Signature:
    """Signature of ``AsyncChatCompletions.create`` in the pinned openai SDK."""
    import openai

    dummy = openai.AsyncOpenAI(api_key="test-key-not-used")
    return inspect.signature(dummy.chat.completions.create)


def _install_anthropic_mock(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Replace ``AnthropicClient._get_client()`` with a mock whose ``messages.create``
    is spec-bound to the real SDK signature. Returns the create-mock so the test can
    assert on its call args."""
    sig = _anthropic_create_signature()

    async def _fake_create(**kwargs: Any) -> Any:
        _bind_or_die(sig, kwargs)
        resp = MagicMock()
        resp.content = [MagicMock(text='{"sub_questions": ["stub"]}')]
        resp.usage = MagicMock(input_tokens=10, output_tokens=5)
        return resp

    create_mock = AsyncMock(side_effect=_fake_create)
    fake_client = MagicMock()
    fake_client.messages.create = create_mock

    def _return_fake(_self: Any) -> Any:
        return fake_client

    monkeypatch.setattr(AnthropicClient, "_get_client", _return_fake)
    return create_mock


def _install_openai_mock(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Same as :func:`_install_anthropic_mock` for the OpenAI client."""
    sig = _openai_create_signature()

    async def _fake_create(**kwargs: Any) -> Any:
        _bind_or_die(sig, kwargs)
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content='{"sub_questions": ["stub"]}'))]
        resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        return resp

    create_mock = AsyncMock(side_effect=_fake_create)
    fake_client = MagicMock()
    fake_client.chat.completions.create = create_mock

    def _return_fake(_self: Any) -> Any:
        return fake_client

    monkeypatch.setattr(OpenAIClient, "_get_client", _return_fake)
    return create_mock


# --------------------------------------------------------------------------------------
# Anthropic — the SDK that broke. Cover every code path that calls it.
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_complete_kwargs_bind_to_the_pinned_sdk_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``AnthropicClient.complete`` must only pass kwargs that bind cleanly to
    the real SDK signature. Regression guard for the SDK 1.x ``temperature``
    removal — a future rename / drop will fail here rather than at first spend."""
    create_mock = _install_anthropic_mock(monkeypatch)
    client = AnthropicClient(api_key="k", model="claude-sonnet-4-6")
    await client.complete(
        [
            Message(role=Role.SYSTEM, content="you are a test"),
            Message(role=Role.USER, content="hi"),
        ],
        max_tokens=64,
        temperature=0.0,
    )
    create_mock.assert_awaited_once()
    call_kwargs = create_mock.await_args.kwargs
    # Positive: temperature must go through extra_body, not as a top-level kwarg
    # (the SDK 1.x removal — the exact fault the hotfix corrects).
    assert "temperature" not in call_kwargs, (
        "temperature must NOT be a top-level kwarg on anthropic 1.x — "
        "route it through extra_body instead"
    )
    assert call_kwargs.get("extra_body") == {"temperature": 0.0}, (
        f"expected extra_body={{'temperature': 0.0}}, got {call_kwargs.get('extra_body')!r}"
    )
    # Positive: mandatory kwargs are all present with the right names.
    assert call_kwargs["model"] == "claude-sonnet-4-6"
    assert call_kwargs["max_tokens"] == 64
    assert call_kwargs["system"] == "you are a test"
    assert call_kwargs["messages"] == [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_anthropic_reachable_from_agent_decomposer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLM point 1/4: agent decompose. Same signature guarantee holds."""
    from verity.agent.decomposer import LLMQuestionDecomposer
    from verity.llm.routing import MultiProviderRoutingClient

    create_mock = _install_anthropic_mock(monkeypatch)
    router = MultiProviderRoutingClient([AnthropicClient(api_key="k")])
    decomposer = LLMQuestionDecomposer(router)
    await decomposer.decompose("what is the notice period?")
    create_mock.assert_awaited_once()
    # No unbound kwarg made it through — the bind assertion in the mock did the work.


@pytest.mark.asyncio
async def test_anthropic_reachable_from_agent_synthesizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLM point 2/4: agent synthesize."""
    from verity.agent.synthesizer import CitedSynthesizer
    from verity.llm.routing import MultiProviderRoutingClient

    create_mock = _install_anthropic_mock(monkeypatch)
    router = MultiProviderRoutingClient([AnthropicClient(api_key="k")])
    synth = CitedSynthesizer(router)
    await synth.synthesize("q", hits=[])
    create_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_anthropic_reachable_from_confidence_scorer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLM point 3/4: agent confidence scorer."""
    from verity.agent.scorer import LLMConfidenceScorer
    from verity.llm.routing import MultiProviderRoutingClient

    create_mock = _install_anthropic_mock(monkeypatch)
    router = MultiProviderRoutingClient([AnthropicClient(api_key="k")])
    scorer = LLMConfidenceScorer(router)
    await scorer.score("q", draft_answer="a", hits=[])
    create_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_anthropic_reachable_from_llm_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM point 4/4: faithfulness judge (the one the eval harness pays for)."""
    from verity.eval.judge import LLMFaithfulnessJudge
    from verity.llm.routing import MultiProviderRoutingClient
    from verity.types import (
        Answer,
        Citation,
        Confidence,
        EvalExample,
        UsageStats,
    )

    create_mock = _install_anthropic_mock(monkeypatch)

    # The mock returns decompose-shaped JSON by default; override for the judge shape.
    async def _judge_response(**kwargs: Any) -> Any:
        sig = _anthropic_create_signature()
        _bind_or_die(sig, kwargs)
        r = MagicMock()
        r.content = [
            MagicMock(text='{"claims":[{"supported":true,"citation_ok":true,"quote":"x"}]}')
        ]
        r.usage = MagicMock(input_tokens=10, output_tokens=5)
        return r

    create_mock.side_effect = _judge_response

    from uuid import uuid4

    router = MultiProviderRoutingClient([AnthropicClient(api_key="k")])
    judge = LLMFaithfulnessJudge(router)
    example = EvalExample(
        id="q-001",
        question="q?",
        answerable=True,
        relevant_chunk_ids=(),
        expected_answer=None,
    )
    answer = Answer(
        query="q?",
        text="an answer with a claim.",
        confidence=Confidence(score=0.9, refused=False, rationale=""),
        hits_used=(),
        citations=(
            Citation(
                chunk_id=str(uuid4()),
                document_id=str(uuid4()),
                quote="x",
                char_start=0,
                char_end=1,
            ),
        ),
        usage=UsageStats(),
    )
    await judge.judge(example, answer)
    create_mock.assert_awaited_once()


# --------------------------------------------------------------------------------------
# OpenAI — pinned 3.x, still exposes temperature. Verify it stays that way.
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_complete_kwargs_bind_to_the_pinned_sdk_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same guarantee for the OpenAI fallback path — its kwargs must bind
    cleanly to the pinned ``chat.completions.create`` signature."""
    create_mock = _install_openai_mock(monkeypatch)
    client = OpenAIClient(api_key="k", model="gpt-4.1-mini")
    await client.complete(
        [
            Message(role=Role.SYSTEM, content="sys"),
            Message(role=Role.USER, content="hi"),
        ],
        max_tokens=64,
        temperature=0.0,
    )
    create_mock.assert_awaited_once()
    call_kwargs = create_mock.await_args.kwargs
    # OpenAI 3.x still exposes temperature as a top-level kwarg.
    assert call_kwargs["temperature"] == 0.0
    assert call_kwargs["max_tokens"] == 64
    assert call_kwargs["model"] == "gpt-4.1-mini"
    assert call_kwargs["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
