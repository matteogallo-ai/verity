"""Property-based tests of the API's input bounds, through FastAPI's TestClient on the
stub agent: question length, Content-Length, upload size and upload type."""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Iterator
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pytest
from fastapi.testclient import TestClient
from hypothesis import assume, given
from hypothesis import strategies as st
from pydantic import SecretStr

from tests.unit.api_app import api_app
from verity.api.guards import UploadRejectedError, check_upload
from verity.config import Settings

_KEY = "test-key-0123456789"
_MAX_UPLOAD = 4096


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    settings = Settings(
        api_keys=(SecretStr(_KEY),),
        api_rate_limit_per_minute=100_000,
        api_max_upload_bytes=_MAX_UPLOAD,
        api_max_documents=10_000,
    )
    with api_app(tmp_path, settings) as test_client:
        test_client.headers["X-API-Key"] = _KEY
        yield test_client


# One character per draw keeps long questions cheap: the bound is about length.
_CHAR = st.sampled_from(["a", " ", "é", "\U0001f600", '"', "\\", "\n"])


@given(_CHAR, st.integers(0, 2_600))
def test_a_question_is_answered_iff_it_fits_the_character_limit(
    client: TestClient, char: str, length: int
) -> None:
    question = char * length
    # json.dumps escapes non-ASCII: the worst case on the wire.
    response = client.post(
        "/ask",
        content=json.dumps({"question": question}),
        headers={"content-type": "application/json"},
    )
    if 1 <= length <= 2_000:
        assert response.status_code == 200, response.text
    elif length == 0:
        assert response.status_code == 422
    else:
        # Past the limit an escaped question can also outgrow the byte budget: 413.
        assert response.status_code in (413, 422)
        assert question[:50] not in response.text, "the rejected input is never echoed"


_NOT_A_LENGTH = st.text(
    alphabet=st.characters(min_codepoint=0x20, max_codepoint=0x7E), max_size=12
).filter(lambda v: not v.isdigit())


@given(_NOT_A_LENGTH, st.sampled_from(["/ask", "/ingest"]))
def test_a_post_without_a_decimal_content_length_is_411(
    client: TestClient, declared: str, path: str
) -> None:
    response = client.post(path, content=b"{}", headers={"content-length": declared})
    assert response.status_code == 411
    assert response.json()["request_id"] == response.headers["X-Request-ID"]


@given(st.integers(0, 10**30), st.sampled_from(["/ask", "/ingest"]))
def test_a_declared_length_over_the_route_limit_is_413_before_parsing(
    client: TestClient, declared: int, path: str
) -> None:
    limit = _MAX_UPLOAD + 16 * 1024 if path == "/ingest" else 2_000 * 12 + 4 * 1024
    assume(declared > limit)
    response = client.post(path, content=b"{}", headers={"content-length": str(declared)})
    assert response.status_code == 413


@given(st.integers(1, 2 * _MAX_UPLOAD), st.sampled_from([".md", ".txt"]))
def test_an_upload_is_stored_iff_it_fits_the_size_limit(
    client: TestClient, size: int, extension: str
) -> None:
    # Distinct bytes per example, so each accepted upload is a new document.
    content = (f"# {size}\n\n".encode() + b"x" * size)[:size]
    response = client.post(
        "/ingest", files={"file": (f"doc{extension}", io.BytesIO(content), "text/plain")}
    )
    if size <= _MAX_UPLOAD:
        assert response.status_code == 200, response.text
        identity = f"upload:/{hashlib.sha256(content).hexdigest()}{extension}"
        assert response.json()["document_id"] == str(uuid5(NAMESPACE_URL, identity))
    else:
        assert response.status_code == 413


# --- type allow-list: extension, declared MIME, content sniffing ------------------------

_TEXT_MIMES = {"text/plain", "text/markdown", "text/x-markdown"}
_ACCEPTED_MIMES = {
    ".md": _TEXT_MIMES,
    ".markdown": _TEXT_MIMES,
    ".txt": {"text/plain"},
    ".pdf": {"application/pdf"},
    ".docx": {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
}
_EXTENSIONS = st.sampled_from([*_ACCEPTED_MIMES, ".exe", ".html", ".md.exe", ""])
_MIMES = st.sampled_from(
    [
        None,
        "application/octet-stream",
        "application/pdf",
        "TEXT/PLAIN; charset=utf-8",
        "text/markdown",
        "text/html",
        "image/png",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ]
)
_CONTENT = st.one_of(
    st.text(max_size=40).map(str.encode),
    st.binary(max_size=40),
    st.binary(max_size=40).map(lambda b: b"%PDF-" + b),
    st.binary(max_size=40).map(lambda b: b"PK\x03\x04" + b),
).filter(bool)


def _acceptable(extension: str, mime: str | None, content: bytes) -> bool:
    if extension not in _ACCEPTED_MIMES:
        return False
    declared = (mime or "application/octet-stream").split(";")[0].strip().lower()
    if declared != "application/octet-stream" and declared not in _ACCEPTED_MIMES[extension]:
        return False
    if extension == ".pdf":
        return content.startswith(b"%PDF-")
    if extension == ".docx":
        return content.startswith(b"PK\x03\x04")
    try:
        content.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return b"\x00" not in content


@given(st.sampled_from(["notes", "Notes", "a.b"]), _EXTENSIONS, st.booleans(), _MIMES, _CONTENT)
def test_check_upload_accepts_exactly_the_allow_listed_triples(
    stem: str, extension: str, upper: bool, mime: str | None, content: bytes
) -> None:
    filename = stem + (extension.upper() if upper else extension)
    try:
        assert check_upload(filename, mime, content) == extension
        accepted = True
    except UploadRejectedError:
        accepted = False
    assert accepted == _acceptable(extension, mime, content)


@given(_EXTENSIONS, _MIMES, _CONTENT)
def test_a_rejected_upload_type_is_415(
    client: TestClient, extension: str, mime: str | None, content: bytes
) -> None:
    # Accepted PDF/DOCX would run docling, which loads models: that path is the slow lane.
    assume(not _acceptable(extension, mime, content))
    response = client.post(
        "/ingest",
        files={"file": (f"upload{extension}", io.BytesIO(content), mime or "")},
    )
    assert response.status_code == 415
