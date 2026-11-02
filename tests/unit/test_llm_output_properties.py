"""Property-based tests of every parser of model output.

Model output is untrusted: it can be prose, truncated JSON, the wrong shape, or text
an injected document dictated. Each parser's contract is to return its typed
"unparseable" value (``None`` or an empty result), never to raise, and never to let a
malformed value through as a valid one.
"""

from __future__ import annotations

import asyncio
import json
import math
from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from verity.agent.decomposer import _parse_sub_questions
from verity.agent.scorer import LLMConfidenceScorer, _parse_confidence
from verity.eval.judge import _parse_judge_output
from verity.llm.clients import StubLLMClient
from verity.llm.json_output import first_matching, parse_json_object
from verity.llm.routing import MultiProviderRoutingClient

_JSON_SCALARS = st.none() | st.booleans() | st.integers() | st.floats() | st.text(max_size=10)
_JSON = st.recursive(
    _JSON_SCALARS,
    lambda inner: (
        st.lists(inner, max_size=4) | st.dictionaries(st.text(max_size=8), inner, max_size=4)
    ),
    max_leaves=12,
)
_KEYS = ["score", "confidence", "refused", "rationale", "claims", "supported", "citation_ok"]
_OBJECTS = st.dictionaries(st.sampled_from(_KEYS) | st.text(max_size=6), _JSON, max_size=5)
# Prose without brackets, so the object is the only JSON in the completion.
_PROSE = st.text(alphabet=st.characters(blacklist_characters="{}[]"), max_size=20)
_DEPTH = st.integers(0, 120_000)


@st.composite
def _completions(draw: st.DrawFn) -> str:
    """What a model returns: valid JSON, wrapped JSON, nested JSON, or any text."""
    kind = draw(st.integers(0, 3))
    if kind == 0:
        return json.dumps(draw(_JSON))
    if kind == 1:
        return draw(_PROSE) + json.dumps(draw(_OBJECTS)) + draw(_PROSE)
    if kind == 2:
        depth = draw(_DEPTH)
        opener, closer = draw(st.sampled_from([("[", "]"), ('{"a":', "}")]))
        return opener * depth + "1" + closer * depth
    return draw(st.text(max_size=80))


@given(_completions())
def test_parse_json_object_returns_a_dict_or_none(text: str) -> None:
    parsed = parse_json_object(text)
    assert parsed is None or isinstance(parsed, dict)


@given(_OBJECTS, _PROSE, _PROSE, st.sampled_from(["{}", "```json\n{}\n```", "[{}]"]))
def test_an_object_survives_prose_fences_and_array_wrapping(
    obj: dict[str, Any], before: str, after: str, wrapper: str
) -> None:
    text = before + wrapper.replace("{}", json.dumps(obj)) + after
    # NaN != NaN, so compare the canonical serialisations.
    assert json.dumps(parse_json_object(text), sort_keys=True) == json.dumps(obj, sort_keys=True)


@given(st.dictionaries(st.sampled_from(_KEYS), st.integers()), st.permutations(_KEYS))
def test_first_matching_returns_the_first_alias_present(
    mapping: dict[str, int], keys: list[str]
) -> None:
    present = [k for k in keys if k in mapping]
    assert first_matching(mapping, tuple(keys)) == (mapping[present[0]] if present else None)


@given(_completions())
def test_confidence_parser_yields_a_finite_unit_score_or_none(text: str) -> None:
    parsed = _parse_confidence(text)
    if parsed is not None:
        score, rationale, refused = parsed
        assert math.isfinite(score) and 0.0 <= score <= 1.0
        assert isinstance(rationale, str) and isinstance(refused, bool)


@given(st.floats(), st.booleans())
def test_a_non_finite_score_is_unparseable_and_refused(score: float, model_refused: bool) -> None:
    text = json.dumps({"score": score, "refused": model_refused, "rationale": "r"})
    scorer = LLMConfidenceScorer(
        MultiProviderRoutingClient([StubLLMClient(lambda _: text)]), threshold=0.55
    )
    confidence = asyncio.run(scorer.score("q?", "answer", hits=[]))
    if not math.isfinite(score):
        assert confidence.refused and confidence.score == 0.0
    else:
        assert confidence.score == max(0.0, min(1.0, score))
        assert confidence.refused == (model_refused or confidence.score < 0.55)


def test_a_nan_score_does_not_pass_the_threshold() -> None:
    """Found by the finite-score property: min(1.0, nan) is 1.0, so NaN shipped answers."""
    assert _parse_confidence('{"score": NaN, "refused": false}') is None


def test_deeply_nested_output_is_unparseable_not_a_crash() -> None:
    """Found by the dict-or-none property: json raised RecursionError past ~10^5 levels."""
    assert parse_json_object("[" * 100_000 + "]" * 100_000) is None
    assert _parse_sub_questions("[" * 100_000 + "]" * 100_000) == []


@given(_completions())
def test_judge_parser_counts_are_bounded_by_its_verdicts(text: str) -> None:
    parsed = _parse_judge_output(text)
    if parsed is not None:
        verdicts, supported, citation_ok, total = parsed
        assert total == len(verdicts) > 0
        assert supported == sum(v.supported for v in verdicts)
        assert citation_ok == sum(v.citation_ok for v in verdicts)


@given(_completions() | st.lists(st.text(max_size=12), max_size=5).map(json.dumps))
def test_sub_questions_are_non_blank_stripped_strings(text: str) -> None:
    for question in _parse_sub_questions(text):
        assert isinstance(question, str) and question and question == question.strip()
