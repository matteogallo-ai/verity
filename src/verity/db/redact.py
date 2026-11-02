"""Strip Postgres credentials from driver messages before they are raised or logged.

libpq quotes the fragment of a connection string it cannot parse, so a malformed
password comes back verbatim in the error (``invalid percent-encoded token:
"<password>"``). Every database error crosses :func:`redact_conninfo` before it
becomes an exception message, a log field or CLI output.
"""

from __future__ import annotations

import re
from urllib.parse import unquote, urlsplit

_MASK = "***"
# user:password@ inside any URL-shaped DSN, not only ours.
_URL_PASSWORD = re.compile(r"(://[^:/@\s\"']*:)[^@\s\"']*@")
# password=value in key-value conninfo; the value may be single-quoted with spaces.
_KV_PASSWORD = re.compile(r"(password\s*=\s*)('(?:[^'\\]|\\.)*'|[^\s'\"]+)", re.IGNORECASE)


def _passwords(database_url: str) -> set[str]:
    found: set[str] = set()
    try:
        url_password = urlsplit(database_url).password
    # urlsplit rejects some malformed netlocs; the regexes still apply.
    except ValueError:
        url_password = None
    if url_password:
        found |= {url_password, unquote(url_password)}
    for match in _KV_PASSWORD.finditer(database_url):
        value = match.group(2)
        found |= {value, value.strip("'")}
    return {p for p in found if p}


def redact_conninfo(message: str, database_url: str) -> str:
    """``message`` with every password from ``database_url`` and every DSN password
    pattern replaced by ``***``.

    A short password that also occurs in ordinary words is masked there too: an
    over-redacted message is the safe failure.
    """
    for secret in sorted(_passwords(database_url), key=len, reverse=True):
        message = message.replace(secret, _MASK)
    message = _URL_PASSWORD.sub(rf"\g<1>{_MASK}@", message)
    return _KV_PASSWORD.sub(rf"\g<1>{_MASK}", message)
