"""Request guards for the HTTP API: upload type checks."""

from __future__ import annotations

from dataclasses import dataclass

# --- uploads ---------------------------------------------------------------------------

# Sent by browsers and curl when the OS has no mapping for the extension (common for
# .md); accepted only because the content is then sniffed below.
_GENERIC_MIME = "application/octet-stream"
_TEXT_MIMES = frozenset({"text/plain", "text/markdown", "text/x-markdown"})
_ALLOWED: dict[str, frozenset[str]] = {
    ".md": _TEXT_MIMES,
    ".markdown": _TEXT_MIMES,
    ".txt": frozenset({"text/plain"}),
    ".pdf": frozenset({"application/pdf"}),
    ".docx": frozenset({"application/vnd.openxmlformats-officedocument.wordprocessingml.document"}),
}
_MAGIC = {".pdf": b"%PDF-", ".docx": b"PK\x03\x04"}
ALLOWED_EXTENSIONS: tuple[str, ...] = tuple(_ALLOWED)


@dataclass(frozen=True)
class UploadRejectedError(Exception):
    message: str


def check_upload(filename: str, content_type: str | None, content: bytes) -> str:
    """The parser extension for an upload, or :class:`UploadRejectedError`.

    Three checks, all required: the extension is allow-listed, the declared MIME type
    matches it (or is the generic octet-stream), and the bytes look like that type
    (magic number for PDF/DOCX, UTF-8 without NUL for text).
    """
    lowered = filename.lower()
    ext = next((e for e in ALLOWED_EXTENSIONS if lowered.endswith(e)), None)
    if ext is None:
        raise UploadRejectedError(f"Unsupported file type; allowed: {', '.join(_ALLOWED)}.")
    mime = (content_type or _GENERIC_MIME).split(";", 1)[0].strip().lower()
    if mime != _GENERIC_MIME and mime not in _ALLOWED[ext]:
        raise UploadRejectedError(f"Content type {mime!r} does not match {ext}.")
    magic = _MAGIC.get(ext)
    if magic is not None:
        if not content.startswith(magic):
            raise UploadRejectedError(f"Content is not a {ext} file.")
    elif b"\x00" in content or not _is_utf8(content):
        raise UploadRejectedError("Text uploads must be UTF-8.")
    return ext


def _is_utf8(content: bytes) -> bool:
    try:
        content.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


__all__ = [
    "ALLOWED_EXTENSIONS",
    "UploadRejectedError",
    "check_upload",
]
