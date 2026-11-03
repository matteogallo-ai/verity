"""A corpus document that is edited or deleted leaves nothing stale in the index.

The auditor's scenario: a clause says "30 days", the file is edited to "90 days" and
ingested again. Only the new text may be retrieved and cited; a citation validated
against a stale chunk would ship an outdated clause with a quote that is no longer in
the document (claim C01).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit.api_app import FakeEmbeddingBackend, FakeRerankerBackend
from verity.agent.synthesizer import _validate_citations
from verity.ingestion import LocalEmbedder
from verity.retrieval import CrossEncoderReranker
from verity.retrieval.hybrid import HybridRetriever
from verity.retrieval.memory_store import InMemoryVectorStore
from verity.runtime import ingest_corpus

_CLAUSE = "## Termination\n\nEither party may terminate this agreement with {} written notice.\n"
_OTHER = "## Fees\n\nInvoices are payable within 45 days of receipt.\n"


class _CountingBackend(FakeEmbeddingBackend):
    def __init__(self) -> None:
        super().__init__(384)
        self.encoded: list[str] = []

    def encode(self, sentences, **kwargs):  # type: ignore[no-untyped-def,override]
        self.encoded.extend(sentences)
        return super().encode(sentences, **kwargs)


def _setup(tmp_path: Path) -> tuple[Path, InMemoryVectorStore, LocalEmbedder, _CountingBackend]:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    backend = _CountingBackend()
    return corpus, InMemoryVectorStore(), LocalEmbedder(backend=backend), backend


def _retriever(store: InMemoryVectorStore, embedder: LocalEmbedder) -> HybridRetriever:
    return HybridRetriever(
        store=store,
        embedder=embedder,
        reranker=CrossEncoderReranker(model="fake", backend=FakeRerankerBackend()),
    )


async def _citable(retriever: HybridRetriever, quote: str) -> list[str]:
    """Every chunk text against which ``quote`` validates as a citation."""
    hits = await retriever.retrieve("termination written notice days", k=8)
    cited = _validate_citations([(quote, i) for i in range(len(hits))], hits)
    return [c.quote for c in cited]


@pytest.mark.parametrize("new", ["90 days'", "ninety (90) days'"], ids=["same-span", "new-span"])
@pytest.mark.asyncio
async def test_an_edited_document_is_only_retrievable_and_citable_in_its_new_text(
    tmp_path: Path, new: str
) -> None:
    corpus, store, embedder, _ = _setup(tmp_path)
    doc = corpus / "msa.md"
    doc.write_text(_CLAUSE.format("30 days'"), encoding="utf-8")
    await ingest_corpus(corpus_dir=corpus, store=store, embedder=embedder)
    retriever = _retriever(store, embedder)
    assert await _citable(retriever, "30 days'")

    doc.write_text(_CLAUSE.format(new), encoding="utf-8")
    await ingest_corpus(corpus_dir=corpus, store=store, embedder=embedder)

    hits = await retriever.retrieve("termination written notice days", k=8)
    assert hits and all("30 days" not in h.chunk.text for h in hits)
    assert await _citable(retriever, "30 days'") == []
    assert await _citable(retriever, new) == [new]
    [summary] = await store.list_documents()
    assert summary.n_chunks == 1


@pytest.mark.asyncio
async def test_a_document_deleted_from_the_corpus_leaves_the_index(tmp_path: Path) -> None:
    corpus, store, embedder, _ = _setup(tmp_path)
    (corpus / "msa.md").write_text(_CLAUSE.format("30 days'"), encoding="utf-8")
    (corpus / "fees.md").write_text(_OTHER, encoding="utf-8")
    await ingest_corpus(corpus_dir=corpus, store=store, embedder=embedder)

    (corpus / "msa.md").unlink()
    await ingest_corpus(corpus_dir=corpus, store=store, embedder=embedder)

    assert [Path(d.uri).name for d in await store.list_documents()] == ["fees.md"]
    hits = await _retriever(store, embedder).retrieve("termination written notice", k=8)
    assert all("terminate" not in h.chunk.text for h in hits)


@pytest.mark.asyncio
async def test_an_unchanged_document_is_not_embedded_again(tmp_path: Path) -> None:
    """The content hash makes a restart over the same corpus free of re-embedding."""
    corpus, store, embedder, backend = _setup(tmp_path)
    (corpus / "msa.md").write_text(_CLAUSE.format("30 days'"), encoding="utf-8")
    (corpus / "fees.md").write_text(_OTHER, encoding="utf-8")
    await ingest_corpus(corpus_dir=corpus, store=store, embedder=embedder)
    first = len(backend.encoded)

    (corpus / "fees.md").write_text(_OTHER.replace("45", "60"), encoding="utf-8")
    await ingest_corpus(corpus_dir=corpus, store=store, embedder=embedder)

    re_embedded = backend.encoded[first:]
    assert re_embedded and all("60 days" in t for t in re_embedded)


@pytest.mark.asyncio
async def test_a_file_that_fails_to_parse_is_not_pruned(tmp_path: Path) -> None:
    """A file still in the corpus keeps its last good version when it fails to parse."""
    corpus, store, embedder, _ = _setup(tmp_path)
    (corpus / "fees.md").write_text(_OTHER, encoding="utf-8")
    (corpus / "notes.pdf").write_bytes(b"%PDF-1.4 not really a pdf")
    await ingest_corpus(corpus_dir=corpus, store=store, embedder=embedder)
    (corpus / "msa.md").write_text(_CLAUSE.format("30 days'"), encoding="utf-8")
    await ingest_corpus(corpus_dir=corpus, store=store, embedder=embedder)
    assert sorted(Path(d.uri).name for d in await store.list_documents()) == ["fees.md", "msa.md"]
