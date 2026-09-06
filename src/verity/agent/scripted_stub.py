"""A scripted stub responder for offline / demo runs of the agent.

Used by ``verity ask --stub``, the S3 integration test, and — critically — by the S4
eval harness in ``--judge stub`` mode. One responder drives all three so the
demo, the ask-e2e test, and the eval scorecard exercise the same code paths.

It's a stub, not a fake win: it knows only the themes the shipped corpus supports.
Questions off-theme produce an honest refusal — that is the *feature* being measured
by the S4 refusal-calibration track, not a bug in the stub.

When a question matches multiple themes on generic keywords (e.g. "notice" appears in
both termination and maintenance topics), the matcher picks the theme with the
highest keyword-overlap score, breaking ties by declaration order.
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
    # MSA
    _Theme(
        # "terminat" is the shared stem for terminate / termination / terminating.
        question_keywords=("terminat", "cancel"),
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
        question_keywords=("assign", "assignment", "competitor"),
        evidence_anchors=("competitor",),
        canned_answer=(
            "Assignment to a competitor is not permitted without the non-assigning "
            "party's prior written consent."
        ),
    ),
    # Financial summary
    _Theme(
        question_keywords=("revenue", "fy2024", "yoy", "year-over-year", "year over year"),
        evidence_anchors=("$482.0 million", "18.4%"),
        canned_answer=("FY2024 total revenue was $482.0 million, an 18.4% increase over FY2023."),
    ),
    # SLA
    _Theme(
        question_keywords=("uptime", "commitment", "credit", "95%", "availability"),
        evidence_anchors=("99.9%",),
        canned_answer=(
            "Provider commits to 99.9% monthly uptime; if uptime falls below 95.0%, "
            "the Customer is entitled to a 50% service credit of that month's fees."
        ),
    ),
    _Theme(
        question_keywords=("priority", "p1", "response time", "support"),
        evidence_anchors=("one (1) hour",),
        canned_answer=(
            "Provider responds to Priority 1 (critical, service unavailable) requests "
            "within one (1) hour."
        ),
    ),
    _Theme(
        question_keywords=("maintenance", "scheduled"),
        evidence_anchors=("forty-eight (48) hours",),
        canned_answer=(
            "Scheduled maintenance does not count against the uptime commitment, and "
            "Provider gives at least forty-eight (48) hours' advance notice."
        ),
    ),
    # DPA
    _Theme(
        question_keywords=("stored", "location", "transfer", "scc", "personal data", "eu-to-us"),
        evidence_anchors=("Standard Contractual Clauses",),
        canned_answer=(
            "Personal data is stored in Frankfurt (EU) and Virginia (US); EU-to-US "
            "transfers are carried out under the European Commission's Standard "
            "Contractual Clauses."
        ),
    ),
    _Theme(
        question_keywords=("retained", "retention", "deletion", "delete", "how long"),
        evidence_anchors=("ninety (90) days",),
        canned_answer=(
            "Provider deletes all personal data processed on the Controller's behalf "
            "within ninety (90) days of termination, unless retention is required by "
            "applicable law."
        ),
    ),
    # Employee handbook
    _Theme(
        question_keywords=("paid time off", "pto", "vacation", "time off", "holiday"),
        evidence_anchors=("twenty-five (25) days",),
        canned_answer=(
            "Employees are entitled to twenty-five (25) days of paid time off per "
            "calendar year, accruing monthly, with up to five (5) unused days carried "
            "over."
        ),
    ),
    _Theme(
        question_keywords=("office", "hybrid", "days per week", "in-person", "onsite"),
        evidence_anchors=("two (2) days per week",),
        canned_answer=(
            "Employees must work from the office at least two (2) days per week, with "
            "core collaboration hours between 10:00 and 16:00 local time."
        ),
    ),
)


def _match_theme(question: str, evidence: str) -> _Theme | None:
    """Pick the theme whose question keywords score highest on the question AND whose
    anchor is present in the retrieved evidence.

    This is the whole point of the refusal test: even though the retriever returns
    something for every query (retrieval never refuses), the agent must refuse when
    the retrieved evidence does not address the question. The stub reproduces that
    behaviour by requiring both a question-topic match AND an evidence-anchor match.

    We score by the number of matching keywords (not just "at least one") so a
    question like "how much notice for scheduled maintenance?" resolves to the
    maintenance theme rather than the termination one that shares the word "notice".
    Ties break by declaration order so behaviour is stable across runs.
    """
    q_lower = question.lower()
    best: tuple[int, _Theme] | None = None
    for theme in _THEMES:
        score = sum(1 for k in theme.question_keywords if k in q_lower)
        if score == 0:
            continue
        if not any(anchor in evidence for anchor in theme.evidence_anchors):
            continue
        if best is None or score > best[0]:
            best = (score, theme)
    return best[1] if best is not None else None


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
