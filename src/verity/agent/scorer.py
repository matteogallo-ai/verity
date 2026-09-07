"""Calibrated confidence + refusal decision.

Wraps ``confidence.prompt`` and owns the single comparison against
``config.confidence_threshold`` that decides whether the agent ships an answer or a
refusal. The scorer *never* decides silently: below-threshold outcomes carry a
concrete rationale (evidence weak / contradictory / out of scope) so the eval
harness can inspect and the UI can display *why*.

Fail-open on parsing: a malformed JSON response from the model is treated as
"unable to verify grounding" → refused with a rationale that says so. This keeps
the agent honest even when the scorer LLM misbehaves.
"""

from __future__ import annotations

import json

from verity.agent.base import ConfidenceScorer
from verity.config import get_settings
from verity.llm.base import RoutingClient
from verity.llm.prompt import load_prompt
from verity.types import Confidence, RetrievalHit, UsageStats


class LLMConfidenceScorer(ConfidenceScorer):
    """Prompt-driven confidence + refusal.

    ``last_usage`` exposes the token/cost accounting for the most recent
    :meth:`score` call so the agent can include it in the aggregated
    :class:`Answer` usage — the ``ConfidenceScorer`` protocol only returns the
    :class:`Confidence` verdict.
    """

    def __init__(
        self,
        client: RoutingClient,
        *,
        threshold: float | None = None,
        prompt_name: str = "confidence",
    ) -> None:
        settings = get_settings()
        self._client = client
        self._threshold = threshold if threshold is not None else settings.confidence_threshold
        self._prompt = load_prompt(prompt_name)
        self.last_usage: UsageStats | None = None

    @property
    def threshold(self) -> float:
        return self._threshold

    async def score(self, question: str, draft_answer: str, hits: list[RetrievalHit]) -> Confidence:
        messages = self._prompt.render(
            question=question,
            answer=draft_answer,
            evidence=_render_evidence(hits),
        )
        completion = await self._client.complete(messages, max_tokens=256, temperature=0.0)
        self.last_usage = completion.usage
        raw = _parse_confidence(completion.text)
        if raw is None:
            return Confidence(
                score=0.0,
                refused=True,
                rationale="Scorer returned malformed output; refusing on principle.",
            )
        score, rationale, model_refused = raw
        # We honour the model's refusal *and* apply the local threshold — either is
        # sufficient reason to refuse. The threshold is the audit-trail decision.
        refused = model_refused or score < self._threshold
        if not rationale:
            rationale = (
                "Confidence below threshold." if refused else "Evidence supports the answer."
            )
        return Confidence(score=score, refused=refused, rationale=rationale)


def _render_evidence(hits: list[RetrievalHit]) -> str:
    lines: list[str] = []
    for i, hit in enumerate(hits):
        section = hit.chunk.section or "-"
        body = hit.chunk.text.strip()
        lines.append(f"[{i}] score={hit.score:.3f} section={section}\n{body}")
    return "\n\n".join(lines) if lines else "(no evidence available)"


# Key aliases the scorer accepts as a defence-in-depth against real-model
# drift. The prompt now shows the exact schema (score/refused/rationale) so
# Sonnet-class models generally comply, but weaker/older/rewritten variants
# occasionally emit natural-sounding keys (e.g. "confidence", "should_refuse",
# "reasoning"). We accept those rather than refusing on principle — the
# scorer's job is to catch WEAK EVIDENCE, not to punish a well-formed answer
# because the model chose a synonym for the top-level key.
_SCORE_KEYS = ("score", "confidence", "confidence_score")
_REFUSED_KEYS = ("refused", "should_refuse", "refuse", "decline")
_RATIONALE_KEYS = ("rationale", "reasoning", "reason", "explanation", "justification")


def _first_matching(mapping: dict[str, object], keys: tuple[str, ...]) -> object | None:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _parse_confidence(text: str) -> tuple[float, str, bool] | None:
    """Extract ``(score, rationale, refused)`` from a scorer LLM completion.

    Robust to (a) prose wrapping the JSON, (b) markdown code fences, (c) a
    one-element array wrapping the object, and (d) common key aliases. Returns
    ``None`` only when the payload truly does not carry a numeric score — the
    only shape the caller cannot recover meaning from.
    """
    stripped = text.strip()
    if not stripped:
        return None
    candidate = _first_json_object(stripped) or stripped
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    # Unwrap ``[{...}]`` — models sometimes wrap the object in a single-element
    # array. Take the first dict; ignore the rest (the prompt asks for one).
    if isinstance(parsed, list):
        parsed = next((item for item in parsed if isinstance(item, dict)), None)
    if not isinstance(parsed, dict):
        return None
    raw_score = _first_matching(parsed, _SCORE_KEYS)
    if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
        return None
    score = max(0.0, min(1.0, float(raw_score)))
    rationale_raw = _first_matching(parsed, _RATIONALE_KEYS)
    rationale = str(rationale_raw).strip() if rationale_raw is not None else ""
    refused_raw = _first_matching(parsed, _REFUSED_KEYS)
    refused = bool(refused_raw) if refused_raw is not None else False
    return score, rationale, refused


def _first_json_object(text: str) -> str:
    """Return the largest ``{...}`` slice in ``text``, or the first ``[...]``
    slice if no braced object is present. The array path handles the case where
    the model wraps the response in ``[{...}]``."""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    array_start = text.find("[")
    array_end = text.rfind("]")
    if array_start != -1 and array_end != -1 and array_end > array_start:
        return text[array_start : array_end + 1]
    return ""


__all__ = ["LLMConfidenceScorer"]
