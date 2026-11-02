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
import subprocess
import sys
from typing import Any

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from verity.agent.decomposer import _parse_sub_questions
from verity.agent.scorer import LLMConfidenceScorer, _parse_confidence
from verity.eval.judge import _parse_judge_output
from verity.llm.clients import StubLLMClient
from verity.llm.json_output import first_matching, load_json, parse_json_object
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


@given(st.integers(0, 200), st.sampled_from([("[", "]"), ('{"a":', "}")]))
@example(64, ("[", "]"))
@example(65, ("[", "]"))
def test_load_json_decodes_up_to_the_nesting_bound_and_rejects_beyond(
    depth: int, brackets: tuple[str, str]
) -> None:
    opener, closer = brackets
    text = opener * depth + "1" + closer * depth
    if depth <= 64:
        assert load_json(text) == json.loads(text)
    else:
        with pytest.raises(ValueError, match="nested deeper"):
            load_json(text)


@given(st.integers(65, 300), st.integers(1, 3))
def test_load_json_bounds_depth_not_size(siblings: int, depth: int) -> None:
    """Many shallow siblings are fine: closing brackets give their level back."""
    text = "[" + ",".join(["[" * depth + "1" + "]" * depth] * siblings) + "]"
    assert load_json(text) == json.loads(text)


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


_ON_A_SMALL_STACK = """
import threading
from verity.agent.decomposer import _parse_sub_questions
from verity.llm.json_output import parse_json_object
deep = "[" * 100_000 + "]" * 100_000
threading.stack_size(512 * 1024)
results = []
worker = threading.Thread(
    target=lambda: results.extend([parse_json_object(deep), _parse_sub_questions(deep)])
)
worker.start()
worker.join()
print(results)
"""


def test_deeply_nested_output_does_not_crash_a_worker_thread() -> None:
    """Found by mutation testing: in a thread with a 512 KiB stack (macOS's default for
    secondary threads) the JSON decoder overflowed the C stack before Python raised
    RecursionError, and the process died (SIGBUS on macOS)."""
    run = subprocess.run(
        [sys.executable, "-c", _ON_A_SMALL_STACK], capture_output=True, text=True, timeout=60
    )
    assert run.returncode == 0, run.stderr[-500:]
    assert run.stdout.strip() == "[None, []]"


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
