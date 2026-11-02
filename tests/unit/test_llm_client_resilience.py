"""Provider clients: explicit timeout/retries, typed error translation, priced usage.

No network: SDK clients are constructed but never called, and SDK exceptions are
built from in-memory ``httpx2`` requests and responses.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import anthropic
import httpx2
import openai
import pytest
from pydantic import SecretStr

from verity.config import ModelPrice, Settings
from verity.llm.base import AuthenticationError, LLMError, ProviderUnavailableError
from verity.llm.clients import AnthropicClient, OpenAIClient, _raise_translated

_REQUEST = httpx2.Request("POST", "https://provider.invalid/v1/messages")


def _status_error(sdk: ModuleType, name: str, status: int) -> Exception:
    response = httpx2.Response(status, request=_REQUEST)
    cls: Any = getattr(sdk, name)
    return cls(f"{name} {status}", response=response, body=None)  # type: ignore[no-any-return]


def _cases(sdk: ModuleType) -> list[tuple[Exception, type[LLMError]]]:
    return [
        (_status_error(sdk, "AuthenticationError", 401), AuthenticationError),
        (_status_error(sdk, "PermissionDeniedError", 403), AuthenticationError),
        (_status_error(sdk, "RateLimitError", 429), ProviderUnavailableError),
        (_status_error(sdk, "InternalServerError", 500), ProviderUnavailableError),
        (_status_error(sdk, "APIStatusError", 529), ProviderUnavailableError),
        (sdk.APITimeoutError(request=_REQUEST), ProviderUnavailableError),
        (sdk.APIConnectionError(request=_REQUEST), ProviderUnavailableError),
        (_status_error(sdk, "BadRequestError", 400), LLMError),
        (_status_error(sdk, "NotFoundError", 404), LLMError),
        (_status_error(sdk, "APIStatusError", 409), LLMError),
        (ValueError("not an SDK error"), LLMError),
    ]


@pytest.mark.parametrize(
    ("sdk", "exc", "expected"),
    [(sdk, exc, expected) for sdk in (anthropic, openai) for exc, expected in _cases(sdk)],
    ids=lambda v: getattr(v, "__name__", type(v).__name__),
)
def test_sdk_errors_translate_by_class(sdk: ModuleType, exc: Exception, expected: type) -> None:
    with pytest.raises(LLMError) as raised:
        _raise_translated(exc, sdk=sdk, provider=sdk.__name__)
    assert type(raised.value) is expected
    assert raised.value.__cause__ is exc


def test_name_lookalike_is_not_mistaken_for_an_auth_error() -> None:
    """The old translation matched substrings of the class name."""

    class AuthenticationTimeoutHelperError(Exception):
        pass

    with pytest.raises(LLMError) as raised:
        _raise_translated(
            AuthenticationTimeoutHelperError("x"), sdk=anthropic, provider="anthropic"
        )
    assert type(raised.value) is LLMError


def test_sdk_clients_get_the_configured_timeout_and_retries() -> None:
    settings = Settings(llm_timeout_s=12.5, llm_max_retries=1)
    a = AnthropicClient(api_key=SecretStr("k"), settings=settings)._get_client()
    o = OpenAIClient(api_key=SecretStr("k"), settings=settings)._get_client()
    assert (a.timeout, a.max_retries) == (12.5, 1)
    assert (o.timeout, o.max_retries) == (12.5, 1)


def test_cost_comes_from_the_configured_price_table() -> None:
    settings = Settings(
        model_prices={"m": ModelPrice(input_usd_per_mtok=2.0, output_usd_per_mtok=8.0)}
    )
    client = OpenAIClient(api_key=SecretStr("k"), model="m", settings=settings)
    assert client.cost_usd(input_tokens=1_000_000, output_tokens=500_000) == pytest.approx(6.0)


def test_default_prices_cover_the_default_models() -> None:
    settings = Settings()
    for model in (settings.anthropic_model, settings.openai_model, settings.judge_model):
        assert model in settings.model_prices


def test_unpriced_model_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="no price configured for model 'mystery'"):
        AnthropicClient(api_key=SecretStr("k"), model="mystery", settings=Settings())
