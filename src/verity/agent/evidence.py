"""Rendering of untrusted text (retrieved chunks, the question, a draft answer) into
the delimited blocks the synthesis, confidence and eval-judge prompts expect.

Retrieved text is data, not instructions: each chunk sits in its own
``<document index="i">`` element inside one ``<evidence>`` element, and the prompts
tell the model to ignore instructions found there. A document must not be able to
close those elements early and write outside them, so every occurrence of a tag name
the prompts use is neutralised in the untrusted text: its ``<`` becomes U+2039 (single
left-pointing angle quotation mark).

The synthesizer and the confidence scorer show the model the same hits in the same
order; the ``index`` is what a citation's ``chunk_index`` refers back to. The scorer
also sees each hit's retrieval score.
"""

from __future__ import annotations

import re

from verity.types import RetrievalHit

NO_EVIDENCE = "<evidence>\n(no evidence available)\n</evidence>"

# Every element name the prompts use to delimit untrusted text. One list for both
# defences: what neutralise makes inert, verity.agent.support flags as an injection.
DELIMITER_TAGS = ("evidence", "document", "question", "draft_answer", "answer", "citations")
_TAG_NAMES = "|".join(DELIMITER_TAGS)
_TAG_RE = re.compile(rf"<(?=\s*/?\s*(?:{_TAG_NAMES})\b)", re.IGNORECASE)


def neutralise(text: str) -> str:
    """``text`` with every opening or closing delimiter tag made inert.

    A citation quote containing such a tag no longer matches its chunk verbatim and is
    dropped; no document in practice contains these XML tag names.
    """
    return _TAG_RE.sub("\u2039", text)


def _attribute(value: str) -> str:
    return neutralise(value).replace('"', "'").replace("\n", " ")


def render_evidence(hits: list[RetrievalHit], *, with_scores: bool) -> str:
    """Render ``hits`` as ``<document>`` elements inside one ``<evidence>`` element.

    ``with_scores=False`` (synthesizer): ``<document index="i" section="...">``.
    ``with_scores=True`` (scorer): ``<document index="i" score="0.000" section="...">``.
    Chunk text is stripped of surrounding whitespace and neutralised; a missing section
    renders as ``-``; an empty hit list renders as :data:`NO_EVIDENCE`.
    """
    if not hits:
        return NO_EVIDENCE
    blocks: list[str] = []
    for i, hit in enumerate(hits):
        score = f' score="{hit.score:.3f}"' if with_scores else ""
        section = _attribute(hit.chunk.section or "-")
        blocks.append(
            f'<document index="{i}"{score} section="{section}">\n'
            f"{neutralise(hit.chunk.text.strip())}\n</document>"
        )
    return "<evidence>\n" + "\n".join(blocks) + "\n</evidence>"


__all__ = ["DELIMITER_TAGS", "NO_EVIDENCE", "neutralise", "render_evidence"]
