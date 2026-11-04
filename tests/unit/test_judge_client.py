"""Which model judges: another family when its key is available, otherwise a Claude
model different from the agent's; never the agent's own model."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from verity.config import Settings
from verity.llm.clients import AnthropicClient
from verity.llm.routing import MultiProviderRoutingClient, create_judge_client
from verity.types import Provider


def _agent(settings: Settings) -> MultiProviderRoutingClient:
    return MultiProviderRoutingClient(
        [AnthropicClient(api_key=SecretStr("k"), model=settings.anthropic_model, settings=settings)]
    )


def _only(client: MultiProviderRoutingClient) -> tuple[Provider, str]:
    (judge,) = client.clients  # no fallback: a fallback would change the judge silently
    return judge.provider, judge.model


def test_auto_picks_the_other_family_when_its_key_is_set() -> None:
    settings = Settings(openai_api_key=SecretStr("sk-test"))
    judge = create_judge_client(agent_client=_agent(settings), settings=settings)
    assert _only(judge) == (Provider.OPENAI, settings.judge_openai_model)


def test_auto_without_another_family_picks_a_different_claude_model() -> None:
    settings = Settings(anthropic_api_key=SecretStr("k"))
    judge = create_judge_client(agent_client=_agent(settings), settings=settings)
    provider, model = _only(judge)
    assert provider is Provider.ANTHROPIC
    assert model == settings.judge_anthropic_model != settings.anthropic_model


def test_a_judge_identical_to_the_agent_is_refused() -> None:
    settings = Settings(judge_provider="anthropic", judge_anthropic_model="claude-sonnet-4-6")
    with pytest.raises(ValueError, match="judge must not be the agent's model"):
        create_judge_client(agent_client=_agent(settings), settings=settings)


def test_every_default_judge_model_is_priced() -> None:
    settings = Settings()
    for model in (settings.judge_anthropic_model, settings.judge_openai_model):
        assert model in settings.model_prices
