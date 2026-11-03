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
    name = f"{hashlib.sha256(content).hexdigest()}{extension}"
    here = document_id_for_uri(f"upload://{name}")
    with contextlib.chdir(tmp_path):
        elsewhere = document_id_for_uri(f"upload://{name}")
    assert here == elsewhere == uuid5(NAMESPACE_URL, f"upload:/{name}")


def test_an_upload_keeps_the_id_it_was_stored_under() -> None:
    """The value main derived from the repo root and the container's /app: changing it
    would make the unique uri column reject every re-upload of a stored document."""
    assert document_id_for_uri("upload://ab.md") == uuid5(NAMESPACE_URL, "upload:/ab.md")


# Small ranges and a shared document make same-document collisions of the
# "{document}:{ordinal}:{start}:{end}" payload likely to be drawn.
_SMALL = st.integers(0, 12)
_KEY = st.tuples(st.just(UUID(int=7)) | st.uuids(), _SMALL, _SMALL, _SMALL)


@given(_KEY, _KEY)
def test_distinct_chunk_keys_get_distinct_ids(
    key_a: tuple[UUID, int, int, int], key_b: tuple[UUID, int, int, int]
) -> None:
    assert (chunk_id_for(*key_a) == chunk_id_for(*key_b)) == (key_a == key_b)


def test_chunk_id_is_stable_across_releases() -> None:
    """Eval labels store chunk ids: the derivation must not drift."""
    assert chunk_id_for(UUID(int=0), 0, 0, 1) == UUID("266de46d-6801-5929-a3bb-faff239d3a3f")
