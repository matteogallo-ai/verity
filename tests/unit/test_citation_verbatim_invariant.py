"""Ongoing guard: every emitted citation's ``quote`` is a **verbatim substring**
of the referenced source document at ``[char_start:char_end]``.

Two guarantees are stacked:

1. :mod:`verity.agent.synthesizer._validate_citations` drops any citation whose
   ``quote`` is not present in the referenced chunk (``chunk.text.find(quote)``
   returns -1 → skip). No paraphrased citation ever ships.
2. The offsets are re-anchored to ``document.text`` via
   ``chunk.char_start + position_of_quote_in_chunk`` — the S1 offset invariant
   guarantees ``document.text[chunk.char_start:chunk.char_end] == chunk.text``,
   so the absolute span is verbatim into the document too.

This test walks the whole shipped 16-question flow with the scripted stub
agent and asserts the substring identity on every emitted citation. Any
future refactor that lets a paraphrased quote through would break this.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from verity.agent.agent import create_default_agent
from verity.agent.scripted_stub import build_stub_router
from verity.ingestion.chunker import StructureAwareChunker
from verity.ingestion.embedder import LocalEmbedder
from verity.ingestion.parsers import create_default_parser
from verity.ingestion.pipeline import IngestionPipeline, discover_sources
from verity.retrieval import CrossEncoderReranker, HybridRetriever, InMemoryVectorStore
from verity.types import Document

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS = REPO_ROOT / "datasets" / "corpus"
DATASET = REPO_ROOT / "datasets" / "eval" / "questions.jsonl"


class _FakeEmbeddingBackend:
    def __init__(self, dim: int = 32) -> None:
        self._dim = dim

    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> list[list[float]]:
        rows: list[list[float]] = []
        for s in sentences:
            v = [0.0] * self._dim
            for tok in s.lower().split():
                v[hash(tok) % self._dim] += 1.0
            n = sum(x * x for x in v) ** 0.5 or 1.0
            rows.append([x / n for x in v])
        return rows

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim


class _FakeRerankerBackend:
    def predict(
        self,
        sentences: Sequence[tuple[str, str]],
        *,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> list[float]:
        return [float(len(set(q.lower().split()) & set(p.lower().split()))) for q, p in sentences]


async def _load_documents_by_id() -> dict[str, Document]:
    """Parse every corpus file so we can verify the DOCUMENT-level substring
    invariant (not just chunk-level). Uses the same parser the pipeline uses."""
    parser = create_default_parser()
    docs = [await parser.parse(uri) for uri in discover_sources(str(CORPUS))]
    return {str(doc.id): doc for doc in docs}


async def _seed_agent():  # type: ignore[no-untyped-def]
    embedder = LocalEmbedder(model="fake", expected_dim=32, backend=_FakeEmbeddingBackend(32))
    store = InMemoryVectorStore()
    pipeline = IngestionPipeline(
        parser=create_default_parser(),
        chunker=StructureAwareChunker(),
        embedder=embedder,
        sink=store,
    )
    result = await pipeline.ingest(discover_sources(str(CORPUS)))
    assert result.failures == []
    retriever = HybridRetriever(
        store=store,
        embedder=embedder,
        reranker=CrossEncoderReranker(model="fake", backend=_FakeRerankerBackend()),
    )
    agent = create_default_agent(retriever=retriever, client=build_stub_router())
    return agent


def _load_answerable_questions() -> list[str]:
    questions: list[str] = []
    with DATASET.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            record = json.loads(raw)
            if record.get("answerable"):
                questions.append(record["question"])
    return questions


@pytest.mark.asyncio
async def test_every_citation_quote_is_a_verbatim_substring_of_source_document() -> None:
    agent = await _seed_agent()
    documents = await _load_documents_by_id()

    n_citations = 0
    for question in _load_answerable_questions():
        answer = await agent.answer(question)
        for citation in answer.citations:
            n_citations += 1
            doc = documents.get(str(citation.document_id))
            assert doc is not None, (
                f"citation references document {citation.document_id} not in the corpus"
            )
            span = doc.text[citation.char_start : citation.char_end]
            assert span == citation.quote, (
                f"citation quote is NOT a verbatim substring of the source document:\n"
                f"  quote  = {citation.quote!r}\n"
                f"  span   = {span!r}\n"
                f"  offsets= [{citation.char_start}:{citation.char_end}]\n"
                f"  doc.uri= {doc.uri}"
            )
    assert n_citations > 0, "no citations emitted across the answerable set — check stub wiring"
