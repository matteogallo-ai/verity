"""Deterministic id derivation for documents and chunks.

Two identities matter and they must not be conflated:

- **Access URI** — how to *read* the source. For a local file, that's an absolute path
  on the current machine; for a web page, the URL. Kept on ``Document.uri`` for
  provenance and citations, but *never* used to derive an id (absolute paths are
  machine-specific and would produce non-portable ids that break in CI or on any
  clone).
- **Identity URI** — the stable, portable string a document is *known by*. For a
  local file, its POSIX path relative to the repo root (e.g. ``datasets/corpus/msa.md``).
  For a web page, the URL as-is (already portable).

``document_id_for_uri`` accepts either — it always normalises through
:func:`identity_for` before hashing. That way every call site is safe by construction
and existing callers do not need to know the distinction.

Re-ingesting the same source must produce the same identifiers so the vector store
upsert stays idempotent and eval labels (``relevant_chunk_ids``) resolve across
machines and CI runs.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse
from uuid import NAMESPACE_URL, UUID, uuid5

from verity.types import ChunkId, DocumentId

_URL_SCHEMES = {"http", "https"}

_repo_root_cache: Path | None = None


def _find_repo_root() -> Path:
    """Walk up from this module until a ``pyproject.toml`` is found.

    Cached because the answer never changes within a process. Falls back to the
    current working directory only if no marker is found — an installed wheel with
    no source tree — in which case identity_for below can still produce a stable
    result for files under cwd.
    """
    global _repo_root_cache
    if _repo_root_cache is not None:
        return _repo_root_cache
    for candidate in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
        if (candidate / "pyproject.toml").is_file():
            _repo_root_cache = candidate
            return candidate
    _repo_root_cache = Path.cwd().resolve()
    return _repo_root_cache


def identity_for(uri: str) -> str:
    """Return the stable identity string derived from an access URI.

    - ``http(s)://`` URLs are returned unchanged.
    - ``file://`` URIs and bare filesystem paths are resolved and expressed as the
      POSIX path *relative to the repo root* — the key portability property. Files
      outside the repo root fall back to their absolute POSIX path so callers still
      get a deterministic identity, just not a portable one (surface this case in
      review rather than silently paper over it).
    """
    parsed = urlparse(uri)
    if parsed.scheme in _URL_SCHEMES:
        return uri
    raw_path = parsed.path if parsed.scheme == "file" else uri
    resolved = Path(raw_path).expanduser().resolve()
    root = _find_repo_root()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()


def document_id_for_uri(uri: str) -> DocumentId:
    """A document's id is uuid5 of its *identity* string under the URL namespace.

    Callers may pass either the access URI or an already-normalised identity — the
    function normalises through :func:`identity_for` either way, so the resulting id
    is always portable across machines and clones.
    """
    return uuid5(NAMESPACE_URL, identity_for(uri))


def chunk_id_for(document_id: DocumentId, ordinal: int, char_start: int, char_end: int) -> ChunkId:
    """A chunk's id captures its document, position, and text span.

    The span (start/end) is part of the key on purpose: if a re-run of the chunker
    produces a different segmentation, the id changes — which is what we want, since
    the chunk is no longer the same retrievable unit.
    """
    payload = f"{document_id}:{ordinal}:{char_start}:{char_end}"
    return uuid5(NAMESPACE_URL, payload)


__all__ = ["UUID", "chunk_id_for", "document_id_for_uri", "identity_for"]
