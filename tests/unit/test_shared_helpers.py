"""Shared helpers extracted from the synthesizer, scorer, judge, agent, CLI and runtime."""

from __future__ import annotations

from verity.llm.clients import StubLLMClient
from verity.llm.routing import MultiProviderRoutingClient, primary_client_identity
from verity.types import Provider


class TestPrimaryClientIdentity:
    def test_reads_first_registered_client(self) -> None:
        router = MultiProviderRoutingClient(
            [
                StubLLMClient(lambda _m: "", model="m-1", provider=Provider.ANTHROPIC),
                StubLLMClient(lambda _m: "", model="m-2", provider=Provider.OPENAI),
            ]
        )
        assert primary_client_identity(router) == (Provider.ANTHROPIC, "m-1")

    def test_object_without_clients_yields_nones(self) -> None:
        assert primary_client_identity(object()) == (None, None)

    def test_untyped_fields_are_dropped(self) -> None:
        class _Loose:
            provider = "anthropic"  # a str, not a Provider
            model = 42

        class _Router:
            clients = (_Loose(),)

        assert primary_client_identity(_Router()) == (None, None)
