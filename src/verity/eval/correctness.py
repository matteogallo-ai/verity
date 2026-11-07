"""Correctness against the dataset's reference answer: an LLM grade, cross-read by code.

The faithfulness judge (:mod:`verity.eval.judge`) asks whether an answer is supported
by the passages retrieved for it; an answer can be faithful to a wrong passage. This
module asks whether it is right, against ``EvalExample.reference_answer``. It grades
committed run artefacts offline (``scripts/judge_correctness.py``): no agent call.

Two signals, published side by side:

- :class:`LLMCorrectnessJudge` renders ``prompts/judge_correctness.prompt`` and parses
  one of ``correct`` / ``partial`` / ``incorrect``. Output it cannot parse is recorded
  as ``unparsed``, never mapped to a verdict.
- :func:`numbers_match` is deterministic: every number of the reference, in digits or
  as a word up to twenty, must appear in the answer. It covers only references with a
  number, and it counts numbers that are not quantities ("24/7", "milestone 2", "one"
  as a pronoun), so it
  does not replace the grade; where both apply, their disagreements are the questions
  to read first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from verity.agent.evidence import neutralise
from verity.llm.base import RoutingClient
from verity.llm.json_output import parse_json_object
from verity.llm.prompt import load_prompt

Verdict = Literal["correct", "partial", "incorrect", "unparsed"]
_VERDICTS: tuple[Verdict, ...] = ("correct", "partial", "incorrect")
# Opus always thinks and bills only the tokens it produces: headroom costs nothing,
# a verdict cut short would parse as unparsed.
MAX_TOKENS = 2048
# A number with optional thousands separators and decimals: 42, 42.00, 1,250,000, 99.9.
_NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")


@dataclass(frozen=True)
class CorrectnessVerdict:
    verdict: Verdict
    reason: str
    cost_usd: float


def parse_verdict(text: str) -> tuple[Verdict, str] | None:
    """``(verdict, reason)`` from the judge's JSON, or ``None`` when it holds none."""
    parsed = parse_json_object(text)
    if parsed is None:
        return None
    raw = parsed.get("verdict")
    if not isinstance(raw, str):
        return None
    verdict = raw.strip().lower()
    for known in _VERDICTS:
        if verdict == known:
            reason = parsed.get("reason")
            return known, reason.strip() if isinstance(reason, str) else ""
    return None


class LLMCorrectnessJudge:
    """Grades one answer against its reference with ``prompts/judge_correctness.prompt``."""

    def __init__(self, client: RoutingClient, *, prompt_name: str = "judge_correctness") -> None:
        self._client = client
        self._prompt = load_prompt(prompt_name)

    async def grade(self, question: str, reference_answer: str, answer: str) -> CorrectnessVerdict:
        messages = self._prompt.render(
            question=question,
            reference_answer=reference_answer,
            answer=neutralise(answer),
        )
        completion = await self._client.complete(messages, max_tokens=MAX_TOKENS)
        parsed = parse_verdict(completion.text)
        if parsed is None:
            return CorrectnessVerdict("unparsed", "", completion.usage.cost_usd)
        return CorrectnessVerdict(parsed[0], parsed[1], completion.usage.cost_usd)


_WORDS = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
]
_WORD_RE = re.compile(r"\b(" + "|".join(_WORDS) + r")\b", re.IGNORECASE)


def _numbers(text: str) -> set[float]:
    """Numbers written in digits, and the number words zero to twenty ("five years")."""
    values = {float(m.replace(",", "")) for m in _NUMBER_RE.findall(text)}
    values |= {float(_WORDS.index(m.lower())) for m in _WORD_RE.findall(text)}
    return values


def numbers_match(reference: str, answer: str) -> bool | None:
    """Whether every number of ``reference`` appears in ``answer`` (42 equals 42.00);
    ``None`` when the reference holds no number, where this check says nothing."""
    expected = _numbers(reference)
    if not expected:
        return None
    return expected <= _numbers(answer)


__all__ = [
    "CorrectnessVerdict",
    "LLMCorrectnessJudge",
    "Verdict",
    "numbers_match",
    "parse_verdict",
]
