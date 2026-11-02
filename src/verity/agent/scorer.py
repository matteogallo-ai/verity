"""Confidence score and its refusal decision (model refusal, or score below the threshold).

Wraps ``confidence.prompt`` and owns the single comparison against
``config.confidence_threshold`` that decides whether the agent ships an answer or a
refusal. The scorer *never* decides silently: below-threshold outcomes carry a
concrete rationale (evidence weak / contradictory / out of scope) so the eval
harness can inspect and the UI can display *why*.

Fail-closed on parsing: a malformed JSON response from the model is treated as
"unable to verify grounding" and refused, with a rationale that says so. The score
is the model's self-report, not a calibrated probability; calibration has not been
measured.
"""

from __future__ import annotations

from verity.agent.base import ConfidenceScorer
from verity.agent.evidence import neutralise, render_evidence
from verity.config import get_settings
from verity.llm.base import RoutingClient
from verity.llm.json_output import first_matching, parse_json_object
from verity.llm.prompt import load_prompt
from verity.types import Confidence, RefusalReason, RetrievalHit, UsageStats


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
            question=neutralise(question),
            answer=neutralise(draft_answer),
            evidence=render_evidence(hits, with_scores=True),
        )
        completion = await self._client.complete(messages, max_tokens=256)
        self.last_usage = completion.usage
        raw = _parse_confidence(completion.text)
        if raw is None:
            return Confidence(
                score=0.0,
                refused=True,
                rationale="Scorer returned malformed output; refusing on principle.",
                reason=RefusalReason.SCORER_UNPARSEABLE,
            )
        score, rationale, model_refused = raw
        # We honour the model's refusal *and* apply the local threshold — either is
        # sufficient reason to refuse. The threshold is the audit-trail decision.
        reason: RefusalReason | None = None
        if model_refused:
            reason = RefusalReason.MODEL_REFUSED
        elif score < self._threshold:
            reason = RefusalReason.BELOW_THRESHOLD
        refused = reason is not None
        if not rationale:
            rationale = (
                "Confidence below threshold." if refused else "Evidence supports the answer."
            )
        return Confidence(score=score, refused=refused, rationale=rationale, reason=reason)


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


def _parse_confidence(text: str) -> tuple[float, str, bool] | None:
    """Extract ``(score, rationale, refused)`` from a scorer LLM completion.

    Robust to (a) prose wrapping the JSON, (b) markdown code fences, (c) a
    one-element array wrapping the object, and (d) common key aliases. Returns
    ``None`` only when the payload truly does not carry a numeric score — the
    only shape the caller cannot recover meaning from.
    """
    parsed = parse_json_object(text)
    if parsed is None:
        return None
    raw_score = first_matching(parsed, _SCORE_KEYS)
    if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
        return None
    score = max(0.0, min(1.0, float(raw_score)))
    rationale_raw = first_matching(parsed, _RATIONALE_KEYS)
    rationale = str(rationale_raw).strip() if rationale_raw is not None else ""
    refused_raw = first_matching(parsed, _REFUSED_KEYS)
    refused = bool(refused_raw) if refused_raw is not None else False
    return score, rationale, refused


__all__ = ["LLMConfidenceScorer"]
