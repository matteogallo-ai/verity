"""Provider clients: explicit timeout/retries, typed error translation, priced usage.

No network: SDK clients are constructed but never called, and SDK exceptions are
built from in-memory ``httpx2`` requests and responses.
"""

from __future__ import annotations

import inspect
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
from verity.types import Message, Role

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
    for model in (settings.anthropic_model, settings.openai_model):
        assert model in settings.model_prices


def test_unpriced_model_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="no price configured for model 'mystery'"):
        AnthropicClient(api_key=SecretStr("k"), model="mystery", settings=Settings())


class _RaisingCreate:
    def __init__(self, error: Exception) -> None:
        self._error = error

    @property
    def with_raw_response(self) -> _RaisingCreate:
        return self

    async def create(self, **_: object) -> object:
        raise self._error


@pytest.mark.asyncio
@pytest.mark.parametrize("client_cls", [AnthropicClient, OpenAIClient])
async def test_a_failing_sdk_call_raises_the_typed_error(client_cls: type[Any]) -> None:
    sdk = anthropic if client_cls is AnthropicClient else openai
    client = client_cls(api_key=SecretStr("k"), settings=Settings())
    failing = _RaisingCreate(_status_error(sdk, "AuthenticationError", 401))
    client._client = (
        type("SDK", (), {"messages": failing})()
        if client_cls is AnthropicClient
        else type("SDK", (), {"chat": type("Chat", (), {"completions": failing})()})()
    )
    with pytest.raises(AuthenticationError):
        await client.complete([Message(role=Role.USER, content="hi")])


@pytest.mark.parametrize(
    ("client_cls", "target"),
    [(AnthropicClient, "anthropic.AsyncAnthropic"), (OpenAIClient, "openai.AsyncOpenAI")],
)
def test_a_client_that_cannot_be_built_is_unavailable(
    client_cls: type[Any], target: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _broken(**_: object) -> None:
        raise ValueError("bad option")

    module, name = target.split(".")
    monkeypatch.setattr(anthropic if module == "anthropic" else openai, name, _broken)
    with pytest.raises(ProviderUnavailableError, match="init failed"):
        client_cls(api_key=SecretStr("k"), settings=Settings())._get_client()


class _Models:
    """``client.models`` whose ``retrieve`` is bound to the pinned SDK's signature."""

    def __init__(self, sdk_models: Any, error: Exception | None = None) -> None:
        self._sig = inspect.signature(sdk_models.retrieve)
        self._error = error
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    async def retrieve(self, *args: object, **kwargs: object) -> object:
        self._sig.bind(*args, **kwargs)
        self.calls.append((args, kwargs))
        if self._error is not None:
            raise self._error
        return object()


def _sdk_models(client_cls: type[Any]) -> Any:
    if client_cls is AnthropicClient:
        return anthropic.AsyncAnthropic(api_key="test-key-not-used").models
    return openai.AsyncOpenAI(api_key="test-key-not-used").models


@pytest.mark.asyncio
@pytest.mark.parametrize("client_cls", [AnthropicClient, OpenAIClient])
async def test_ping_asks_the_free_models_endpoint_for_the_configured_model(
    client_cls: type[Any],
) -> None:
    client = client_cls(api_key=SecretStr("k"), settings=Settings())
    models = _Models(_sdk_models(client_cls))
    client._client = type("SDK", (), {"models": models})()
    await client.ping()
    assert models.calls == [((client.model,), {})]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("client_cls", "error_name", "status", "expected"),
    [
        (AnthropicClient, "AuthenticationError", 401, AuthenticationError),
        (OpenAIClient, "InternalServerError", 503, ProviderUnavailableError),
    ],
)
async def test_a_failing_ping_raises_the_typed_error(
    client_cls: type[Any], error_name: str, status: int, expected: type[Exception]
) -> None:
    sdk = anthropic if client_cls is AnthropicClient else openai
    client = client_cls(api_key=SecretStr("k"), settings=Settings())
    models = _Models(_sdk_models(client_cls), _status_error(sdk, error_name, status))
    client._client = type("SDK", (), {"models": models})()
    with pytest.raises(expected):
        await client.ping()


_ANTHROPIC_OK = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-6",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 3, "output_tokens": 1},
}
_OPENAI_OK = {
    "id": "c1",
    "object": "chat.completion",
    "created": 0,
    "model": "gpt-4.1-mini",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
}


@pytest.mark.asyncio
@pytest.mark.parametrize("client_cls", [AnthropicClient, OpenAIClient])
async def test_a_call_reports_the_retries_the_sdk_made(client_cls: type[Any]) -> None:
    """SDK retries are invisible in the latency alone; a run of record whose p95 was
    blown by retried calls must say so."""
    import httpx

    # Each SDK ships its own HTTP library; the mock transport must come from it.
    http: Any = httpx2 if client_cls is AnthropicClient else httpx
    replies = iter([503, 200])

    def handler(_request: Any) -> Any:
        status = next(replies)
        if status != 200:
            # retry-after-ms keeps the SDK's backoff to a millisecond.
            return http.Response(status, json={}, headers={"retry-after-ms": "1"})
        return http.Response(
            200, json=_ANTHROPIC_OK if client_cls is AnthropicClient else _OPENAI_OK
        )

    sdk_cls = anthropic.AsyncAnthropic if client_cls is AnthropicClient else openai.AsyncOpenAI
    client = client_cls(api_key=SecretStr("k"), settings=Settings())
    client._client = sdk_cls(
        api_key="k",
        max_retries=2,
        http_client=http.AsyncClient(transport=http.MockTransport(handler)),
    )
    completion = await client.complete([Message(role=Role.USER, content="hi")])
    assert completion.text == "ok"
    assert completion.usage.retries == 1


def test_an_answer_sums_the_retries_of_its_calls() -> None:
    from verity.agent.agent import _aggregate_usage
    from verity.types import UsageStats

    calls = [UsageStats(llm_calls=1, retries=r) for r in (0, 2, 1)]
    assert _aggregate_usage(*calls, None, latency_ms=1.0).retries == 3
