"""Shared helpers extracted from the synthesizer, scorer, judge, agent, CLI and runtime.

The evidence-rendering tests pin the exact bytes both prompts received before the
extraction: a change here changes what the model sees, so it must be deliberate.
"""

from __future__ import annotations

import pytest

from verity.agent.evidence import NO_EVIDENCE, render_evidence
from verity.llm.clients import StubLLMClient
from verity.llm.json_output import first_json_object, first_matching, parse_json_object
from verity.llm.routing import MultiProviderRoutingClient, primary_client_identity
from verity.types import Chunk, Provider, RetrievalHit, RetrieverKind


def _hit(text: str, *, section: str | None, score: float, ordinal: int) -> RetrievalHit:
    chunk = Chunk(
        document_id="00000000-0000-0000-0000-000000000000",
        text=text,
        ordinal=ordinal,
        char_start=0,
        char_end=len(text),
        section=section,
    )
    return RetrievalHit(chunk=chunk, score=score, kind=RetrieverKind.RERANKED, rank=ordinal)


HITS = [
    _hit("  Either party may terminate.\n", section="1. Term", score=0.8234, ordinal=0),
    _hit("Fees are due in 30 days.", section=None, score=0.1, ordinal=1),
]


class TestRenderEvidence:
    def test_synthesizer_format_is_unchanged(self) -> None:
        assert render_evidence(HITS, with_scores=False) == (
            "[0] (section: 1. Term)\nEither party may terminate.\n\n"
            "[1] (section: -)\nFees are due in 30 days."
        )

    def test_scorer_format_is_unchanged(self) -> None:
        assert render_evidence(HITS, with_scores=True) == (
            "[0] score=0.823 section=1. Term\nEither party may terminate.\n\n"
            "[1] score=0.100 section=-\nFees are due in 30 days."
        )

    @pytest.mark.parametrize("with_scores", [True, False])
    def test_empty_hits_render_placeholder(self, with_scores: bool) -> None:
        assert render_evidence([], with_scores=with_scores) == NO_EVIDENCE


class TestJsonOutput:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ('{"a": 1}', '{"a": 1}'),
            ('Sure! ```json\n{"a": {"b": 2}}\n``` done', '{"a": {"b": 2}}'),
            ('[{"a": 1}]', '{"a": 1}'),
            ('["x", "y"]', '["x", "y"]'),
            ("no json here", ""),
            ("} before {", ""),
        ],
    )
    def test_first_json_object(self, text: str, expected: str) -> None:
        assert first_json_object(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ('{"score": 0.5}', {"score": 0.5}),
            ('prose {"score": 0.5} prose', {"score": 0.5}),
            ('["a", {"score": 0.5}]', {"score": 0.5}),
        ],
    )
    def test_parse_json_object_recovers_drifted_shapes(
        self, text: str, expected: dict[str, float]
    ) -> None:
        assert parse_json_object(text) == expected

    @pytest.mark.parametrize(
        "text", ["", "   ", "not json", "{broken", '["a", "b"]', "42", '{"a":1} {"b":2}']
    )
    def test_parse_json_object_returns_none_when_unrecoverable(self, text: str) -> None:
        assert parse_json_object(text) is None

    def test_array_of_several_objects_is_unrecoverable(self) -> None:
        """The brace slice spans both objects, which is not valid JSON. Callers then
        fail closed (scorer, synthesizer) or mark the example unsupported (judge);
        prompts ask for exactly one object."""
        assert parse_json_object('[{"score": 0.5}, {"score": 0.9}]') is None

    def test_first_matching_respects_key_order(self) -> None:
        mapping = {"confidence": 0.2, "score": 0.9}
        assert first_matching(mapping, ("score", "confidence")) == 0.9
        assert first_matching(mapping, ("missing",)) is None


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
