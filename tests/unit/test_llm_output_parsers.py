"""Realistic-shape parser tests for the 3 LLM output parsers.

Motivation — the S7 live run showed the confidence scorer refusing 16/16
questions because its parser was strict about key names (``score`` /
``refused`` / ``rationale``) while the prompt did not actually specify a
schema in the rendered user block (the schema was in a ``#`` comment that
:mod:`verity.llm.prompt` strips). Sonnet-class models default to natural key
names (``confidence`` / ``should_refuse`` / ``reasoning``) so every parse
returned ``None`` → "malformed output, refusing on principle" → 16/16
refusals → judge_cost=$0 → published numbers meaningless.

This module fixes the class of bug: for every parser in the LLM-output path,
feed the realistic shapes we know real Sonnet-family models emit and assert
they extract meaningful values. If a future prompt change drifts the schema,
these tests fail before another live-spend run.

Zero LLM cost, zero key.
"""

from __future__ import annotations

from verity.agent.decomposer import _parse_sub_questions
from verity.agent.scorer import _parse_confidence
from verity.agent.synthesizer import _parse_synth_output
from verity.eval.judge import _parse_judge_output

# --------------------------------------------------------------------------------------
# Confidence scorer — the parser that failed 16/16 in live run of record.
# --------------------------------------------------------------------------------------


class TestScorerParserRealShapes:
    """Every shape Sonnet 4.6 plausibly emits given the current prompt."""

    def test_perfect_schema_shape(self) -> None:
        """Happy path: the exact keys the prompt now requests."""
        result = _parse_confidence('{"score": 0.85, "refused": false, "rationale": "grounded"}')
        assert result == (0.85, "grounded", False)

    def test_natural_keys_used_by_undirected_sonnet(self) -> None:
        """The exact fault of the broken live run: model emits ``confidence`` /
        ``should_refuse`` / ``reasoning`` when the prompt does not pin the schema.
        The parser must recover the values rather than refusing on principle."""
        result = _parse_confidence(
            '{"confidence": 0.85, "should_refuse": false, "reasoning": "grounded"}'
        )
        assert result is not None
        score, rationale, refused = result
        assert score == 0.85
        assert rationale == "grounded"
        assert refused is False

    def test_confidence_score_key_variant(self) -> None:
        """Another common Sonnet spelling — the ``_score`` suffix."""
        result = _parse_confidence(
            '{"confidence_score": 0.9, "refuse": false, "explanation": "clear"}'
        )
        assert result == (0.9, "clear", False)

    def test_prose_prefix_with_json(self) -> None:
        """Model volunteers a chatty preamble before the JSON — common on the
        first calls before the prompt has settled."""
        text = 'Based on the evidence:\n\n{"confidence": 0.4, "should_refuse": true, "reasoning": "weak"}'
        result = _parse_confidence(text)
        assert result == (0.4, "weak", True)

    def test_markdown_code_fence_wrap(self) -> None:
        text = '```json\n{"score": 0.85, "refused": false, "rationale": "ok"}\n```'
        assert _parse_confidence(text) == (0.85, "ok", False)

    def test_single_element_array_wrap(self) -> None:
        """Model wraps the object in ``[{...}]`` — a common Sonnet quirk when the
        prompt says "the schema" without saying "one object"."""
        text = '[{"score": 0.85, "refused": false, "rationale": "ok"}]'
        assert _parse_confidence(text) == (0.85, "ok", False)

    def test_truly_malformed_still_returns_none(self) -> None:
        """Belt-and-braces: a payload with NO numeric field must still fail so
        the scorer refuses on principle. Tolerance must not silently pass a
        garbage response."""
        assert _parse_confidence("The evidence is weak.") is None
        assert _parse_confidence("") is None
        assert _parse_confidence('{"only": "prose"}') is None


# --------------------------------------------------------------------------------------
# Synthesizer — prompt already has the schema, but the parser must still be
# resilient to the same real-shape drifts.
# --------------------------------------------------------------------------------------


class TestSynthParserRealShapes:
    def test_perfect_shape(self) -> None:
        text = '{"answer": "The notice period is 30 days.", "citations": [{"quote": "30 days", "chunk_index": 0}]}'
        answer, citations = _parse_synth_output(text)
        assert answer == "The notice period is 30 days."
        assert citations == [("30 days", 0)]

    def test_prose_wrapped(self) -> None:
        text = 'Here is my answer:\n\n{"answer": "Yes.", "citations": []}'
        answer, citations = _parse_synth_output(text)
        assert answer == "Yes."
        assert citations == []

    def test_markdown_fenced(self) -> None:
        text = '```json\n{"answer": "OK", "citations": []}\n```'
        assert _parse_synth_output(text) == ("OK", [])

    def test_single_element_array_wrap(self) -> None:
        text = '[{"answer": "hello", "citations": []}]'
        assert _parse_synth_output(text) == ("hello", [])

    def test_no_json_returns_text_verbatim(self) -> None:
        """Fallback shape: no JSON in the output → treat the whole thing as the
        answer with no citations. Citation validation will drop empty citations
        and the scorer will refuse — safe path."""
        text = "I cannot answer this."
        answer, citations = _parse_synth_output(text)
        assert answer == text
        assert citations == []


# --------------------------------------------------------------------------------------
# Decomposer — parser is already fail-open (caller falls back to ``[question]``),
# but assert the accepted shapes still work.
# --------------------------------------------------------------------------------------


class TestDecomposerParserRealShapes:
    def test_perfect_array(self) -> None:
        assert _parse_sub_questions('["one", "two"]') == ["one", "two"]

    def test_single_string_array(self) -> None:
        assert _parse_sub_questions('["only question"]') == ["only question"]

    def test_prose_wrapped_array(self) -> None:
        text = 'Here are the sub-questions:\n\n["a", "b"]'
        assert _parse_sub_questions(text) == ["a", "b"]

    def test_markdown_fenced_array(self) -> None:
        text = '```json\n["a"]\n```'
        assert _parse_sub_questions(text) == ["a"]

    def test_non_list_returns_empty(self) -> None:
        """A non-list JSON should not crash the caller — return ``[]`` so the
        agent falls back to ``[question]``."""
        assert _parse_sub_questions('{"unexpected": "shape"}') == []


# --------------------------------------------------------------------------------------
# Judge — same schema-drift risk (the prompt now specifies keys, parser now
# accepts aliases as defence in depth).
# --------------------------------------------------------------------------------------


class TestJudgeParserRealShapes:
    def test_perfect_shape(self) -> None:
        text = '{"claims": [{"claim": "x", "supported": true, "citation_ok": true}]}'
        result = _parse_judge_output(text)
        assert result is not None
        _, supported, citation_ok, total = result
        assert (supported, citation_ok, total) == (1, 1, 1)

    def test_alias_atomic_claims(self) -> None:
        text = (
            '{"atomic_claims": [{"claim": "x", "is_supported": true, "citation_supports": true}]}'
        )
        result = _parse_judge_output(text)
        assert result is not None
        _, supported, citation_ok, total = result
        assert (supported, citation_ok, total) == (1, 1, 1)

    def test_mixed_support(self) -> None:
        text = (
            '{"claims": ['
            '{"claim": "a", "supported": true, "citation_ok": true},'
            '{"claim": "b", "supported": false, "citation_ok": false},'
            '{"claim": "c", "supported": true, "citation_ok": false}'
            "]}"
        )
        result = _parse_judge_output(text)
        assert result is not None
        _, supported, citation_ok, total = result
        assert (supported, citation_ok, total) == (2, 1, 3)

    def test_empty_claims_list_returns_none(self) -> None:
        """No claims → the judge can't score → return None so the caller
        surfaces the failure honestly rather than reporting 0/0."""
        assert _parse_judge_output('{"claims": []}') is None

    def test_prose_wrapped_and_fenced(self) -> None:
        text = 'Assessment:\n\n```json\n{"claims": [{"claim":"x","supported":true,"citation_ok":true}]}\n```'
        result = _parse_judge_output(text)
        assert result is not None
        _, supported, citation_ok, total = result
        assert (supported, citation_ok, total) == (1, 1, 1)


# --------------------------------------------------------------------------------------
# End-to-end guardrail: the exact fault-mode of the broken run must never
# happen again on the exact same input shape.
# --------------------------------------------------------------------------------------


def test_regression_broken_run_shape_now_parses_successfully() -> None:
    """The literal shape that failed 16/16 in the live run of record. Bind
    this test to the incident: it must PASS on every commit, or a re-spend is
    a re-crash risk. Do not weaken this test — if a future refactor makes it
    fail, the parser lost a real-world tolerance the live run depends on."""
    # The exact class of output Sonnet 4.6 produces when told "return the
    # specified JSON" without being handed the schema.
    text = '{"confidence": 0.85, "should_refuse": false, "reasoning": "grounded in evidence"}'
    result = _parse_confidence(text)
    assert result is not None, (
        "Broken-run regression: parser must accept ``confidence``/``should_refuse``"
        "/``reasoning`` aliases so a schema-less prompt can't silently refuse 100%."
    )
    score, rationale, refused = result
    assert score == 0.85
    assert refused is False
    assert rationale == "grounded in evidence"
