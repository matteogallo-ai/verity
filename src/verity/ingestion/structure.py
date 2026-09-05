"""Structural metadata carried from parser to chunker.

The Document contract is intentionally minimal (`text` + free-form `metadata`). To let
the chunker segment along semantic boundaries without inventing a per-format detour,
parsers serialise a compact list of *sections* into ``metadata["structure"]``.

A section is a half-open span ``[char_start, char_end)`` into ``document.text`` plus
optional heading path and page. The chunker reads this list, packs adjacent sections
into token-budgeted chunks, and — crucially — never crosses a section boundary in a
way that would break the ``document.text[cs:ce] == chunk.text`` invariant.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field


class Section(BaseModel, frozen=True):
    """One structural region of a parsed document.

    Spans are half-open into ``document.text``. ``heading_path`` is a materialised path
    like ``["1. Term", "1.2 Termination"]`` so the chunker can attach it as
    ``chunk.section``. ``page`` is preserved when the parser knows it (PDF).
    """

    char_start: int
    char_end: int
    heading_path: tuple[str, ...] = ()
    page: int | None = None


class DocumentStructure(BaseModel, frozen=True):
    """The full section list for a document. Serialised as JSON into
    ``document.metadata['structure']`` because the frozen Document only stores
    string-valued metadata."""

    sections: tuple[Section, ...] = Field(default_factory=tuple)

    def to_metadata(self) -> dict[str, str]:
        return {"structure": self.model_dump_json()}

    @classmethod
    def from_metadata(cls, metadata: dict[str, str]) -> DocumentStructure:
        raw = metadata.get("structure")
        if raw is None:
            return cls()
        data = json.loads(raw)
        return cls.model_validate(data)


__all__ = ["DocumentStructure", "Section"]
