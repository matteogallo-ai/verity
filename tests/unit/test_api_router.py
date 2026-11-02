"""HTTP endpoints — /status, /ingest, /ask. Deterministic, no keys, no DB.

We rebuild the runtime with the fake embedder + fake reranker from the harness
tests so no torch model is loaded, then swap the stub agent's response in for
a controlled path. The wrapper endpoints must remain a thin serializer — every
assertion here checks the SHAPE of the response mirrors the domain type.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit.api_app import api_app


@pytest.fixture
def client(tmp_path: Path):  # type: ignore[no-untyped-def]
    """Stub-mode app on the shipped corpus."""
    with api_app(tmp_path) as test_client:
        yield test_client


def test_status_reports_real_provenance_and_corpus(client) -> None:  # type: ignore[no-untyped-def]
    r = client.get("/status")
    assert r.status_code == 200
    body = r.json()
    # Real provenance (mode is 'stub' by boot config, no LLM key configured).
    assert body["provenance"]["is_stub"] is True
    assert body["provenance"]["agent_model"] == "stub-agent"
    assert body["provenance"]["embedding_model"]
    assert body["provenance"]["reranker_model"]
    # Corpus preloaded from datasets/corpus (5 documents shipped with the repo).
    assert body["corpus"]["n_documents"] >= 5
    assert body["corpus"]["n_chunks"] >= 10
    assert isinstance(body["corpus"]["sources"], list)
    # Never a hard-coded status string.
    assert "status" not in body or body.get("status") not in {"ok", "OK"}


def test_ask_provenance_reflects_the_actual_provider_used(client) -> None:  # type: ignore[no-untyped-def]
    """Provenance is derived per-request from the synthesizer's Completion.

    In stub mode the synthesizer's completion carries ``Provider.LOCAL``, so
    the API MUST report ``is_stub=True`` + ``agent_model="stub-agent"``. This
    test guards against the "static-provenance-lies-on-fallback" class of
    bug, where the API reported a boot-time model instead of the one used.
    """
    r = client.post(
        "/ask",
        json={"question": "What is the notice period for terminating the agreement?"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    ans = body["answer"]
    prov = body["provenance"]
    # AnswerView carries the raw provider stamp from the synthesizer's Completion.
    assert ans["provider_used"] == "local"
    # `stub` is the model id StubLLMClient carries by default.
    assert ans["model_used"] == "stub"
    # Provenance derives is_stub / agent_model from that stamp — not from boot.
    assert prov["is_stub"] is True
    assert prov["agent_model"] == "stub-agent"


def test_ask_returns_full_answer_view_with_provenance(client) -> None:  # type: ignore[no-untyped-def]
    r = client.post(
        "/ask", json={"question": "What is the notice period for terminating the agreement?"}
    )
    assert r.status_code == 200, r.text
    body = r.json()

    # Provenance is always adjacent to the answer — never separated on the wire.
    assert body["provenance"]["is_stub"] is True
    assert body["provenance"]["agent_model"] == "stub-agent"

    ans = body["answer"]
    assert ans["query"].startswith("What is the notice period")
    assert isinstance(ans["text"], str) and ans["text"]

    # A cited answer carries ≥1 citation whose quote is a substring of a chunk in hits_used.
    citations = ans["citations"]
    hits = ans["hits_used"]
    assert citations, "answerable question should produce a citation"
    hit_ids = {h["chunk_id"] for h in hits}
    for c in citations:
        assert c["chunk_id"] in hit_ids, (
            f"citation refers to chunk {c['chunk_id']} not present in hits_used"
        )

    # Every chunk carries a `cited` flag derived from citations set-membership.
    cited_chunk_ids = {c["chunk_id"] for c in citations}
    for h in hits:
        assert h["cited"] == (h["chunk_id"] in cited_chunk_ids)

    # Confidence surfaced verbatim, refused=False for an answerable question.
    assert ans["confidence"]["refused"] is False
    assert 0.0 <= ans["confidence"]["score"] <= 1.0

    # Usage & trace id are real.
    assert ans["usage"]["latency_ms"] > 0
    assert ans["usage"]["llm_calls"] >= 1
    assert ans["trace_id"]


def test_ask_refusal_state_is_first_class(client) -> None:  # type: ignore[no-untyped-def]
    r = client.post("/ask", json={"question": "What is the CEO's home address?"})
    assert r.status_code == 200, r.text
    body = r.json()
    ans = body["answer"]
    # Refusal is a state, not an error.
    assert ans["confidence"]["refused"] is True
    assert ans["confidence"]["rationale"]
    assert ans["citations"] == []  # no fabricated citations on refusal
    # hits_used is preserved for traceability even on refusal.
    assert isinstance(ans["hits_used"], list)
    # Provenance still adjacent to the answer.
    assert body["provenance"]["is_stub"] is True


def test_ask_rejects_empty_question(client) -> None:  # type: ignore[no-untyped-def]
    r = client.post("/ask", json={"question": ""})
    assert r.status_code == 422  # pydantic validation error, not a stack trace


def test_ingest_upload_uses_content_addressed_id(client) -> None:  # type: ignore[no-untyped-def]
    """Same bytes → same document_id, regardless of the client-supplied filename.

    Guards the upload path against the non-portable-id bug, where ids depended
    on the machine-specific path of the file.
    """
    content = b"# Uploaded document\n\nThis body is content-addressed.\n"
    r1 = client.post(
        "/ingest",
        files={"file": ("first.md", content, "text/markdown")},
    )
    assert r1.status_code == 200, r1.text
    doc_id_1 = r1.json()["document_id"]
    uri_1 = r1.json()["uri"]

    # Re-upload the SAME bytes under a DIFFERENT filename.
    r2 = client.post(
        "/ingest",
        files={"file": ("renamed.md", content, "text/markdown")},
    )
    assert r2.status_code == 200
    doc_id_2 = r2.json()["document_id"]
    uri_2 = r2.json()["uri"]

    # Content-addressed: same bytes → same URI (pure hash, no filename in uri)
    # → same document_id (uuid5(NAMESPACE_URL, uri)). Re-uploads are idempotent.
    assert doc_id_1 == doc_id_2, (
        f"expected content-addressed id to be stable across filenames; got {doc_id_1} vs {doc_id_2}"
    )
    assert uri_1 == uri_2, f"URI should not include filename; got {uri_1!r} vs {uri_2!r}"
    assert uri_1.startswith("upload://")

    # The status endpoint reflects the uploaded document exactly once.
    status = client.get("/status").json()
    matches = [s for s in status["corpus"]["sources"] if s == uri_1]
    assert len(matches) == 1, (
        f"expected exactly one source entry per unique content hash, got {matches}"
    )


def test_ingest_rejects_empty_upload(client) -> None:  # type: ignore[no-untyped-def]
    r = client.post("/ingest", files={"file": ("empty.md", b"", "text/markdown")})
    assert r.status_code == 400
    assert "empty" in r.json()["detail"].lower()


def _rewire_agent(client, *, llm=None, store=None) -> None:  # type: ignore[no-untyped-def]
    from verity.agent.agent import create_default_agent
    from verity.agent.scripted_stub import build_stub_router
    from verity.retrieval import HybridRetriever

    runtime = client.app.state.runtime
    retriever = runtime.retriever
    if store is not None:
        retriever = HybridRetriever(store=store, embedder=runtime.embedder, reranker=None)
    runtime.agent = create_default_agent(retriever=retriever, client=llm or build_stub_router())


def test_ask_answers_503_when_every_provider_is_down(client) -> None:  # type: ignore[no-untyped-def]
    from verity.llm.base import ProviderUnavailableError
    from verity.llm.clients import StubLLMClient
    from verity.llm.routing import MultiProviderRoutingClient

    def _down(_messages):  # type: ignore[no-untyped-def]
        raise ProviderUnavailableError("anthropic: 529 overloaded; openai: timeout")

    _rewire_agent(client, llm=MultiProviderRoutingClient([StubLLMClient(_down)]))
    r = client.post("/ask", json={"question": "What is the notice period?"})
    assert r.status_code == 503
    body = r.json()["detail"]
    assert body["error"] == "provider_unavailable"
    assert body["trace_id"]
    assert "overloaded" not in r.text, "provider detail must stay in the logs"


def test_ask_answers_503_when_the_store_is_down(client) -> None:  # type: ignore[no-untyped-def]
    from verity.retrieval import InMemoryVectorStore

    class _DownStore(InMemoryVectorStore):
        async def dense_search(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise ConnectionError("could not connect to postgres at db:5432")

    client.get("/status")  # preload before swapping the store
    _rewire_agent(client, store=_DownStore())
    r = client.post("/ask", json={"question": "What is the notice period?"})
    assert r.status_code == 503
    assert r.json()["detail"]["error"] == "retrieval_unavailable"
    assert "postgres" not in r.text
