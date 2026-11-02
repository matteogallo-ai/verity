"""Property-based tests of content-addressed and path-derived ids.

Ids key the vector-store upsert and the eval labels, so they must not depend on the
machine, the working directory or the order of calls.
"""

from __future__ import annotations

import contextlib
import hashlib
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

from hypothesis import given
from hypothesis import strategies as st

from verity.ingestion.ids import _find_repo_root, chunk_id_for, document_id_for_uri

_SEGMENT = st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-", min_size=1, max_size=10)
_REL_PATH = st.lists(_SEGMENT, min_size=1, max_size=4).map(lambda parts: "/".join(parts) + ".md")
_SPAN = st.tuples(st.integers(0, 10_000), st.integers(0, 10_000), st.integers(0, 10_000))


@given(_REL_PATH)
def test_a_repo_file_has_the_same_id_as_path_or_file_uri(relative: str) -> None:
    absolute = _find_repo_root() / relative
    by_path = document_id_for_uri(str(absolute))
    assert by_path == document_id_for_uri(absolute.as_uri())
    # Portable: the id is that of the repo-relative path, whatever the checkout location.
    assert by_path == uuid5(NAMESPACE_URL, relative)


@given(_REL_PATH)
def test_a_web_url_is_its_own_identity(path: str) -> None:
    url = f"https://example.com/{path}"
    assert document_id_for_uri(url) == uuid5(NAMESPACE_URL, url)


@given(st.binary(min_size=1, max_size=64), st.sampled_from([".md", ".txt", ".pdf", ".docx"]))
def test_an_upload_id_does_not_depend_on_the_working_directory(
    tmp_path: Path, content: bytes, extension: str
) -> None:
    uri = f"upload://{hashlib.sha256(content).hexdigest()}{extension}"
    here = document_id_for_uri(uri)
    with contextlib.chdir(tmp_path):
        elsewhere = document_id_for_uri(uri)
    assert here == elsewhere == uuid5(NAMESPACE_URL, uri)


@given(st.uuids(), st.integers(0, 1_000), _SPAN)
def test_chunk_id_is_a_function_of_document_ordinal_and_span(
    document_id: UUID, ordinal: int, span: tuple[int, int, int]
) -> None:
    start, end, _ = span
    assert chunk_id_for(document_id, ordinal, start, end) == chunk_id_for(
        document_id, ordinal, start, end
    )


@given(st.uuids(), st.integers(0, 1_000), _SPAN, st.uuids(), st.integers(0, 1_000), _SPAN)
def test_distinct_chunk_keys_get_distinct_ids(
    doc_a: UUID,
    ordinal_a: int,
    span_a: tuple[int, int, int],
    doc_b: UUID,
    ordinal_b: int,
    span_b: tuple[int, int, int],
) -> None:
    key_a = (doc_a, ordinal_a, span_a[0], span_a[1])
    key_b = (doc_b, ordinal_b, span_b[0], span_b[1])
    assert (chunk_id_for(*key_a) == chunk_id_for(*key_b)) == (key_a == key_b)


def test_chunk_id_is_stable_across_releases() -> None:
    """Eval labels store chunk ids: the derivation must not drift."""
    assert chunk_id_for(UUID(int=0), 0, 0, 1) == UUID("266de46d-6801-5929-a3bb-faff239d3a3f")
