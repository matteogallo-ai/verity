"""A scripted stub responder for offline / demo runs of the agent.

Used by ``verity ask --stub`` and by the integration test — the same script drives
both so a reviewer can see, on their laptop with no keys, the exact behaviour the
CI test asserts. It's a stub, not a fake win: it can only answer the four themes
the shipped corpus supports (termination notice, liability cap, FY2024 revenue,
assignment to competitor). Out-of-scope questions produce an honest refusal — that
is the feature, not a bug in the stub.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from verity.llm.clients import Responder, StubLLMClient
from verity.llm.routing import MultiProviderRoutingClient
from verity.types import Message


@dataclass(frozen=True)
class _Theme:
    """One demo topic: keywords for question-topic detection + evidence anchor phrases."""

    question_keywords: tuple[str, ...]
    evidence_anchors: tuple[str, ...]  # tried in order; first one present in evidence wins
    canned_answer: str


_THEMES: tuple[_Theme, ...] = (
    _Theme(
        question_keywords=("terminate", "termination", "notice", "cancel"),
        evidence_anchors=("thirty (30) days",),
        canned_answer=(
            "Either party may terminate for convenience with thirty (30) days' prior "
            "written notice."
        ),
    ),
    _Theme(
        question_keywords=("liable", "liability", "cap", "data-processing", "breach"),
        evidence_anchors=("one million", "$1,000,000"),
        canned_answer=(
            "Provider's aggregate liability is capped at one million United States "
            "dollars ($1,000,000) or the fees paid over the preceding 12 months, "
            "whichever is greater."
        ),
    ),
    _Theme(
        question_keywords=("revenue", "fy2024", "yoy", "year-over-year", "year over year"),
        evidence_anchors=("$482.0 million", "18.4%"),
        canned_answer=("FY2024 total revenue was $482.0 million, an 18.4% increase over FY2023."),
    ),
    _Theme(
        question_keywords=("assign", "assignment", "competitor"),
        evidence_anchors=("competitor",),
        canned_answer=(
            "Assignment to a competitor is not permitted without the non-assigning "
            "party's prior written consent."
        ),
    ),
)


def _match_theme(question: str, evidence: str) -> _Theme | None:
    """Only answer when the *question* is on-topic for a theme whose anchor is in evidence.

    This is the whole point of the refusal test: even though the retriever returns
    something for every query (retrieval never refuses), the agent must refuse when
    the retrieved evidence does not address the question. The stub reproduces that
    behaviour by requiring both a question-topic match AND an evidence-anchor match.
    """
    q_lower = question.lower()
    for theme in _THEMES:
        if not any(k in q_lower for k in theme.question_keywords):
            continue
        if any(anchor in evidence for anchor in theme.evidence_anchors):
            return theme
    return None


def _classify(messages: list[Message]) -> str:
    system = (messages[0].content if messages else "").lower()
    if "decompose" in system:
        return "decompose"
    if "assess whether" in system:
        return "confidence"
    if "concise" in system or "drafts" in system:
        return "synthesize"
    return "unknown"


def _index_containing(evidence: str, needle: str) -> int:
    idx = 0
    for line in evidence.splitlines():
        if line.startswith("[") and "]" in line:
            try:
                idx = int(line[1 : line.find("]")])
            except ValueError:
                continue
        elif needle in line:
            return idx
    return 0


_QUESTION_LINE_RE = re.compile(r"^\s*(QUESTION|Question)\s*:\s*(.*?)\s*$", re.MULTILINE)


def _extract_question(user_content: str) -> str:
    """Recover the original question from the rendered user prompt of any of our prompts."""
    match = _QUESTION_LINE_RE.search(user_content)
    return match.group(2).strip() if match else user_content.strip()


def scripted_responder() -> Responder:
    """Build the responder used by ``verity ask --stub`` and the integration tests."""

    def _responder(messages: list[Message]) -> str:
        kind = _classify(messages)
        user_content = messages[-1].content if messages else ""
        question = _extract_question(user_content)

        if kind == "decompose":
            return json.dumps([question])
        if kind == "synthesize":
            evidence = user_content
            theme = _match_theme(question, evidence)
            if theme is None:
                return json.dumps(
                    {
                        "answer": "The supplied evidence does not answer this question.",
                        "citations": [],
                    }
                )
            anchor = next(a for a in theme.evidence_anchors if a in evidence)
            return json.dumps(
                {
                    "answer": theme.canned_answer,
                    "citations": [
                        {"quote": anchor, "chunk_index": _index_containing(evidence, anchor)}
                    ],
                }
            )
        if kind == "confidence":
            evidence = user_content
            theme = _match_theme(question, evidence)
            if theme is not None:
                return json.dumps(
                    {
                        "score": 0.88,
                        "refused": False,
                        "rationale": "Answer is grounded in a verbatim clause from the retrieved evidence.",
                    }
                )
            return json.dumps(
                {
                    "score": 0.15,
                    "refused": True,
                    "rationale": "The retrieved evidence does not address the question — refusing rather than guessing.",
                }
            )
        return "{}"

    return _responder


def build_stub_router() -> MultiProviderRoutingClient:
    """Convenience: a RoutingClient with a single scripted stub client."""
    return MultiProviderRoutingClient([StubLLMClient(scripted_responder())])


__all__ = ["build_stub_router", "scripted_responder"]
