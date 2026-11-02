"""API hardening: bounded inputs, typed uploads, generic errors, API keys, rate limit."""

from __future__ import annotations

import io
from pathlib import Path

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


def test_rate_limiter_window_slides() -> None:
    now = [0.0]
    limiter = RateLimiter(limit=2, window_s=60.0, clock=lambda: now[0])
    assert limiter.hit("k") is None
    assert limiter.hit("k") is None
    retry = limiter.hit("k")
    assert retry is not None and 59.0 < retry <= 60.0
    now[0] = 60.5
    assert limiter.hit("k") is None


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
