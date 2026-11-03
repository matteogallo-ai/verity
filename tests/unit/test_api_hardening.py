"""API hardening: bounded inputs, typed uploads, generic errors, API keys, rate limit."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from tests.unit.api_app import api_app
from verity.api import create_app
from verity.api.guards import RateLimiter
from verity.config import Settings

_KEY = "test-key-0123456789"
_QUESTION = "What is the notice period for terminating the agreement?"


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {"api_keys": (SecretStr(_KEY),)}
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


@pytest.fixture
def client(tmp_path: Path):  # type: ignore[no-untyped-def]
    with api_app(tmp_path, _settings()) as test_client:
        test_client.headers["X-API-Key"] = _KEY
        yield test_client


def _upload_with_key(client: TestClient, name: str, content: bytes, mime: str):  # type: ignore[no-untyped-def]
    return client.post(
        "/ingest", files={"file": (name, io.BytesIO(content), mime)}, headers={"X-API-Key": _KEY}
    )


def _upload(client: TestClient, name: str, content: bytes, mime: str):  # type: ignore[no-untyped-def]
    return client.post("/ingest", files={"file": (name, io.BytesIO(content), mime)})


# --- startup ---------------------------------------------------------------------------


def test_app_refuses_to_start_without_keys_unless_auth_is_disabled(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="VERITY_API_KEYS"):
        create_app(runs_dir=tmp_path, agent_mode="stub", settings=Settings())


# --- authentication --------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/ask", "/ingest"])
def test_missing_key_is_401(client: TestClient, path: str) -> None:
    del client.headers["X-API-Key"]
    r = client.post(path, json={"question": _QUESTION})
    assert r.status_code == 401


def test_wrong_key_is_401(client: TestClient) -> None:
    r = client.post("/ask", json={"question": _QUESTION}, headers={"X-API-Key": "nope"})
    assert r.status_code == 401


def test_status_needs_a_key(client: TestClient) -> None:
    """/status lists every document URI and the model provenance."""
    del client.headers["X-API-Key"]
    assert client.get("/status").status_code == 401
    assert client.get("/status", headers={"X-API-Key": _KEY}).status_code == 200


def test_health_stays_public(client: TestClient) -> None:
    del client.headers["X-API-Key"]
    assert client.get("/health").status_code == 200


def test_valid_key_is_served(client: TestClient) -> None:
    assert client.post("/ask", json={"question": _QUESTION}).status_code == 200


def test_auth_can_be_disabled_explicitly_for_dev(tmp_path: Path) -> None:
    with api_app(tmp_path, Settings(api_auth_disabled=True)) as anonymous:
        assert anonymous.post("/ask", json={"question": _QUESTION}).status_code == 200


# --- rate limit ------------------------------------------------------------------------


def test_rate_limit_is_per_key_and_answers_429_with_retry_after(tmp_path: Path) -> None:
    settings = _settings(
        api_keys=(SecretStr(_KEY), SecretStr("second-key-0123456789")),
        api_rate_limit_per_minute=2,
    )
    with api_app(tmp_path, settings) as c:
        c.headers["X-API-Key"] = _KEY
        assert c.post("/ask", json={"question": _QUESTION}).status_code == 200
        assert c.post("/ask", json={"question": _QUESTION}).status_code == 200
        limited = c.post("/ask", json={"question": _QUESTION})
        assert limited.status_code == 429
        assert int(limited.headers["Retry-After"]) >= 1
        other = c.post(
            "/ask", json={"question": _QUESTION}, headers={"X-API-Key": "second-key-0123456789"}
        )
        assert other.status_code == 200


def test_status_counts_toward_the_per_key_rate_limit(tmp_path: Path) -> None:
    with api_app(tmp_path, _settings(api_rate_limit_per_minute=2)) as c:
        c.headers["X-API-Key"] = _KEY
        assert c.get("/status").status_code == 200
        assert c.get("/status").status_code == 200
        assert c.post("/ask", json={"question": _QUESTION}).status_code == 429


def test_rate_limiter_window_slides() -> None:
    now = [0.0]
    limiter = RateLimiter(limit=2, window_s=60.0, clock=lambda: now[0])
    assert limiter.hit("k") is None
    assert limiter.hit("k") is None
    retry = limiter.hit("k")
    assert retry is not None and 59.0 < retry <= 60.0
    now[0] = 60.5
    assert limiter.hit("k") is None


def test_failed_key_attempts_are_rate_limited_per_client(tmp_path: Path) -> None:
    with api_app(tmp_path, _settings(api_auth_failures_per_minute=3)) as c:
        for _ in range(3):
            assert c.get("/status", headers={"X-API-Key": "guess"}).status_code == 401
        blocked = c.get("/status", headers={"X-API-Key": "guess"})
        assert blocked.status_code == 429
        assert int(blocked.headers["Retry-After"]) >= 1
        # Checked before the key: a correct guess from a blocked client learns nothing.
        assert c.get("/status", headers={"X-API-Key": _KEY}).status_code == 429


def test_successful_requests_do_not_count_as_failed_attempts(tmp_path: Path) -> None:
    with api_app(tmp_path, _settings(api_auth_failures_per_minute=2)) as c:
        c.headers["X-API-Key"] = _KEY
        for _ in range(5):
            assert c.get("/status").status_code == 200


def test_rate_limiter_peek_does_not_record_a_hit() -> None:
    limiter = RateLimiter(limit=1, window_s=60.0, clock=lambda: 0.0)
    assert limiter.blocked("ip") is None
    assert limiter.hit("ip") is None
    assert limiter.blocked("ip") is not None


def test_rate_limiter_memory_is_bounded() -> None:
    """Failed attempts are keyed by client address, which a client chooses."""
    now = [0.0]
    limiter = RateLimiter(limit=1, window_s=60.0, clock=lambda: now[0], max_principals=100)
    for n in range(1000):
        limiter.hit(f"10.0.{n // 256}.{n % 256}")
    assert len(limiter) <= 100
    now[0] = 61.0
    limiter.hit("10.9.9.9")
    assert len(limiter) == 1


def test_eviction_drops_expired_buckets_before_a_live_one() -> None:
    now = [0.0]
    limiter = RateLimiter(limit=1, window_s=60.0, clock=lambda: now[0], max_principals=3)
    limiter.hit("old-1")
    limiter.hit("old-2")
    now[0] = 50.0
    limiter.hit("blocked")
    now[0] = 70.0
    limiter.hit("new")
    assert limiter.blocked("blocked") is not None


@pytest.mark.parametrize(
    ("first", "second", "shared"),
    [
        ("2001:db8::1", "2001:db8::ffff:1", True),
        ("2001:db8::1", "2001:db8:0:1::1", False),
        ("192.0.2.1", "192.0.2.2", False),
        ("::ffff:192.0.2.1", "192.0.2.1", True),
        ("testclient", "testclient", True),
    ],
)
def test_failed_attempts_are_counted_per_ipv6_64(first: str, second: str, shared: bool) -> None:
    """One IPv6 host usually holds a whole /64: counted per address, it would get ten
    attempts per address."""
    from verity.api.guards import client_bucket

    assert (client_bucket(first) == client_bucket(second)) is shared


# --- spending budget -------------------------------------------------------------------


def _bill_every_answer(client: TestClient, cost_usd: float, *, error: bool = False) -> None:
    """Make every stub answer report ``cost_usd``; with ``error``, also fail it."""
    from verity.types import AnswerError, ErrorKind

    runtime = client.app.state.runtime  # type: ignore[attr-defined]
    real_ask = runtime.ask

    async def _billed(question: str):  # type: ignore[no-untyped-def]
        answer = await real_ask(question)
        update: dict[str, object] = {
            "usage": answer.usage.model_copy(update={"cost_usd": cost_usd})
        }
        if error:
            update |= {
                "confidence": None,
                "error": AnswerError(kind=ErrorKind.PROVIDER_UNAVAILABLE, detail="down"),
            }
        return answer.model_copy(update=update)

    runtime.ask = _billed


def test_spending_budget_is_per_key_and_answers_429_with_retry_after(tmp_path: Path) -> None:
    second = "second-key-0123456789"
    settings = _settings(api_keys=(SecretStr(_KEY), SecretStr(second)), api_budget_usd=0.05)
    with api_app(tmp_path, settings) as c:
        _bill_every_answer(c, 0.03)
        c.headers["X-API-Key"] = _KEY
        assert c.post("/ask", json={"question": _QUESTION}).status_code == 200
        # $0.03 spent is under $0.05: admitted, and this answer takes the key over.
        assert c.post("/ask", json={"question": _QUESTION}).status_code == 200
        limited = c.post("/ask", json={"question": _QUESTION})
        assert limited.status_code == 429
        assert limited.json()["detail"] == "Spending budget exceeded."
        assert 1 <= int(limited.headers["Retry-After"]) <= 3600
        other = c.post("/ask", json={"question": _QUESTION}, headers={"X-API-Key": second})
        assert other.status_code == 200


def test_an_answer_that_errors_after_spending_counts_against_the_budget(
    tmp_path: Path,
) -> None:
    with api_app(tmp_path, _settings(api_budget_usd=0.05)) as c:
        _bill_every_answer(c, 0.06, error=True)
        c.headers["X-API-Key"] = _KEY
        assert c.post("/ask", json={"question": _QUESTION}).status_code == 503
        assert c.post("/ask", json={"question": _QUESTION}).status_code == 429


def test_spend_budget_window_slides() -> None:
    from verity.api.guards import SpendBudget

    now = [0.0]
    budget = SpendBudget(budget_usd=1.0, window_s=3600.0, clock=lambda: now[0])
    assert budget.exhausted("k") is None
    budget.record("k", 0.6)
    now[0] = 100.0
    budget.record("k", 0.6)
    retry = budget.exhausted("k")
    # Under budget again once the first $0.60 leaves the window, at t=3600.
    assert retry is not None and retry == pytest.approx(3500.0)
    assert budget.exhausted("other") is None
    now[0] = 3600.5
    assert budget.exhausted("k") is None


def test_spend_budget_retry_waits_for_as_many_entries_as_needed() -> None:
    from verity.api.guards import SpendBudget

    now = [0.0]
    budget = SpendBudget(budget_usd=1.0, window_s=3600.0, clock=lambda: now[0])
    for at in (0.0, 10.0, 20.0):
        now[0] = at
        budget.record("k", 0.5)
    # $1.50 spent: under $1.00 only once the first two entries have expired, at t=3610.
    assert budget.exhausted("k") == pytest.approx(3590.0)


def test_spend_budget_rejects_a_non_positive_budget() -> None:
    from verity.api.guards import SpendBudget

    with pytest.raises(ValueError, match="positive"):
        SpendBudget(budget_usd=0.0, window_s=60.0)


# --- input bounds ----------------------------------------------------------------------


def test_question_longer_than_the_limit_is_422(client: TestClient) -> None:
    r = client.post("/ask", json={"question": "x" * 2001})
    assert r.status_code == 422
    assert "x" * 100 not in r.text, "the rejected input must not be echoed back"


def test_oversized_request_body_is_413_before_parsing(client: TestClient) -> None:
    r = client.post("/ask", content=b"{" + b" " * 200_000 + b"}")
    assert r.status_code == 413


def test_two_megabyte_upload_is_413(client: TestClient) -> None:
    r = _upload(client, "big.txt", b"a" * (2 * 1024 * 1024), "text/plain")
    assert r.status_code == 413


@pytest.mark.parametrize(
    ("name", "content", "mime"),
    [
        ("tool.exe", b"MZ\x90\x00", "application/x-msdownload"),
        ("notes", b"no extension", "text/plain"),
        ("notes.md", b"# fine", "image/png"),  # declared type contradicts the extension
        ("fake.pdf", b"not a pdf at all", "application/pdf"),  # magic bytes missing
        ("fake.docx", b"plain text", "application/octet-stream"),
        ("binary.txt", b"\x00\x01\x02\xff", "text/plain"),  # not UTF-8 text
    ],
    ids=["exe", "no-extension", "mime-mismatch", "pdf-magic", "docx-magic", "binary-text"],
)
def test_disallowed_upload_type_is_415(
    client: TestClient, name: str, content: bytes, mime: str
) -> None:
    assert _upload(client, name, content, mime).status_code == 415


def test_octet_stream_markdown_is_accepted_after_sniffing(client: TestClient) -> None:
    r = _upload(client, "note.md", b"# Title\n\nSome text.", "application/octet-stream")
    assert r.status_code == 200, r.text


def test_corpus_document_limit_is_409(tmp_path: Path) -> None:
    with api_app(tmp_path, _settings(api_max_documents=7)) as c:
        c.headers["X-API-Key"] = _KEY
        assert _upload(c, "a.txt", b"first upload", "text/plain").status_code == 200
        assert _upload(c, "b.txt", b"second upload", "text/plain").status_code == 409
        # Re-uploading the same bytes is idempotent and does not count twice.
        assert _upload(c, "a.txt", b"first upload", "text/plain").status_code == 200


# --- error bodies ----------------------------------------------------------------------


def test_corrupt_pdf_is_a_generic_422_with_a_request_id(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _explode(**_kw: object) -> None:
        raise RuntimeError(
            "Conversion failed for: tmpx38dtysn.pdf\n  File /site-packages/docling/x.py"
        )

    monkeypatch.setattr(client.app.state.runtime, "ingest_bytes", _explode)  # type: ignore[attr-defined]
    r = _upload(client, "broken.pdf", b"%PDF-1.4 garbage", "application/pdf")
    assert r.status_code == 422
    body = r.json()
    assert "tmp" not in r.text and "docling" not in r.text and "File " not in r.text
    assert body["request_id"] == r.headers["X-Request-ID"]


def test_unhandled_error_is_a_generic_500_with_a_request_id(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _explode(_question: str) -> None:
        raise KeyError("internal detail /srv/secret/path")

    monkeypatch.setattr(client.app.state.runtime, "ask", _explode)  # type: ignore[attr-defined]
    r = client.post("/ask", json={"question": _QUESTION})
    assert r.status_code == 500
    assert "secret" not in r.text
    assert r.json()["request_id"] == r.headers["X-Request-ID"]


def test_every_response_carries_a_request_id(client: TestClient) -> None:
    assert client.get("/health").headers["X-Request-ID"]
    echoed = client.get("/health", headers={"X-Request-ID": "abc-123"})
    assert echoed.headers["X-Request-ID"] == "abc-123"
    forged = client.get("/health", headers={"X-Request-ID": "x" * 500 + "\n"})
    assert forged.headers["X-Request-ID"] != "x" * 500


# --- store outage ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [("get", "/status", None), ("post", "/ask", {"question": _QUESTION})],
)
def test_unreachable_postgres_is_a_generic_503(
    tmp_path: Path, method: str, path: str, body: dict[str, str] | None
) -> None:
    import psycopg

    refused = psycopg.OperationalError("connection to server at 10.0.0.1 port 5432 failed")
    with (
        patch("psycopg.AsyncConnection.connect", side_effect=refused),
        api_app(tmp_path, _settings(), store="pgvector") as down,
    ):
        response = down.request(method, path, json=body, headers={"X-API-Key": _KEY})

    assert response.status_code == 503
    assert response.json()["detail"]["error"] == "retrieval_unavailable"
    assert "10.0.0.1" not in response.text
    assert response.json()["request_id"]


@pytest.mark.parametrize(
    ("error", "status"),
    [
        ("unavailable", 503),
        ("schema", 503),
    ],
)
def test_a_store_failure_on_ingest_is_not_blamed_on_the_document(
    tmp_path: Path, error: str, status: int
) -> None:
    from verity.retrieval.base import StoreUnavailableError
    from verity.retrieval.pgvector_store import EmbeddingDimensionMismatchError

    exc = (
        StoreUnavailableError("cannot connect to Postgres")
        if error == "unavailable"
        else EmbeddingDimensionMismatchError("schema holds VECTOR(384), configured 768")
    )
    with api_app(tmp_path, _settings()) as client:
        client.get("/status")
        runtime = client.app.state.runtime  # type: ignore[attr-defined]
        with patch.object(runtime.store, "list_documents", side_effect=exc):
            response = _upload_with_key(client, "note.md", b"# Note\n\nText.\n", "text/markdown")

    assert response.status_code == status
    assert "parsed" not in response.text
    assert "VECTOR" not in response.text and "Postgres" not in response.text


def test_a_question_at_the_limit_fits_the_body_even_when_json_escaped(
    client: TestClient,
) -> None:
    """Python's json.dumps escapes non-ASCII: an astral character is 12 bytes on the
    wire. A question within the character limit must reach the length check."""
    import json

    body = json.dumps({"question": "\U0001f600" * 2000})  # 2000 characters, 24 KB
    response = client.post("/ask", content=body, headers={"content-type": "application/json"})
    assert response.status_code != 413
