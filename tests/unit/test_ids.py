"""Deterministic ids: same input → same UUID, always."""

from __future__ import annotations

from verity.ingestion.ids import chunk_id_for, document_id_for_uri


def test_document_id_is_deterministic() -> None:
    a = document_id_for_uri("file:///corpus/msa.md")
    b = document_id_for_uri("file:///corpus/msa.md")
    assert a == b


def test_document_id_changes_with_uri() -> None:
    a = document_id_for_uri("file:///corpus/msa.md")
    b = document_id_for_uri("file:///corpus/msa2.md")
    assert a != b


def test_chunk_id_is_deterministic() -> None:
    doc_id = document_id_for_uri("file:///corpus/msa.md")
    a = chunk_id_for(doc_id, ordinal=0, char_start=10, char_end=42)
    b = chunk_id_for(doc_id, ordinal=0, char_start=10, char_end=42)
    assert a == b


def test_chunk_id_varies_with_span() -> None:
    doc_id = document_id_for_uri("file:///corpus/msa.md")
    a = chunk_id_for(doc_id, ordinal=0, char_start=10, char_end=42)
    b = chunk_id_for(doc_id, ordinal=0, char_start=10, char_end=43)
    c = chunk_id_for(doc_id, ordinal=1, char_start=10, char_end=42)
    assert len({a, b, c}) == 3
