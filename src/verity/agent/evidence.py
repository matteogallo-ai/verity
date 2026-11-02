"""Rendering of retrieved hits into the numbered evidence block of a prompt.

The synthesizer and the confidence scorer show the model the same hits in the
same order; the ``[i]`` index is what a citation's ``chunk_index`` refers back to,
so both renderings must number hits identically. They differ only in the header
line: the scorer also sees each hit's retrieval score.
"""

from __future__ import annotations

from verity.types import RetrievalHit

NO_EVIDENCE = "(no evidence available)"


def render_evidence(hits: list[RetrievalHit], *, with_scores: bool) -> str:
    """Render ``hits`` as ``[i]`` blocks separated by blank lines.

    ``with_scores=False`` (synthesizer): ``[i] (section: <section>)``.
    ``with_scores=True`` (scorer): ``[i] score=<0.000> section=<section>``.
    Chunk text is stripped of surrounding whitespace; a missing section renders
    as ``-``; an empty hit list renders as :data:`NO_EVIDENCE`.
    """
    blocks: list[str] = []
    for i, hit in enumerate(hits):
        section = hit.chunk.section or "-"
        header = (
            f"[{i}] score={hit.score:.3f} section={section}"
            if with_scores
            else f"[{i}] (section: {section})"
        )
        blocks.append(f"{header}\n{hit.chunk.text.strip()}")
    return "\n\n".join(blocks) if blocks else NO_EVIDENCE


__all__ = ["NO_EVIDENCE", "render_evidence"]
