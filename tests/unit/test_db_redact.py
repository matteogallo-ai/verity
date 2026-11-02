"""No Postgres password reaches an error message, a traceback or a log line.

libpq echoes fragments of a connection string it cannot parse (an invalid
percent-encoded password comes back verbatim), and the CLI prints tracebacks. Every
case below fails while parsing the connection string, before any socket is opened.
"""

from __future__ import annotations

import traceback

import pytest
from typer.testing import CliRunner

from verity.cli import app
from verity.config import get_settings
from verity.db.redact import redact_conninfo
from verity.retrieval import PgVectorStore
from verity.retrieval.base import StoreUnavailableError

_PASSWORD = "S3cret%zzPw"
_URL = f"postgresql://verity:{_PASSWORD}@localhost:55432/verity"


@pytest.mark.parametrize(
    ("message", "url"),
    [
        (f'invalid percent-encoded token: "{_PASSWORD}"', _URL),
        (f"cannot use {_URL}", _URL),
        ("bad password=hunter2 x", "host=h password=hunter2 user=u"),
        ("bad password = 'hun ter2' x", "host=h password = 'hun ter2' user=u"),
        ("token p@ss:word", "postgresql://u:p%40ss%3Aword@h/db"),
        ("other dsn postgresql://a:zzz@h/db", _URL),
    ],
)
def test_redact_removes_the_password_in_every_form(message: str, url: str) -> None:
    redacted = redact_conninfo(message, url)
    for secret in ("S3cret", "hunter2", "hun ter2", "p@ss:word", "zzz"):
        assert secret not in redacted, redacted


def test_redact_keeps_the_rest_of_the_message() -> None:
    message = 'connection to server at "127.0.0.1", port 5432 failed: Connection refused'
    assert redact_conninfo(message, _URL) == message


@pytest.mark.asyncio
async def test_store_connection_error_carries_no_password() -> None:
    store = PgVectorStore(database_url=_URL)
    with pytest.raises(StoreUnavailableError) as caught:
        await store.dense_search([0.0] * 384, 1)
    rendered = "".join(traceback.format_exception(caught.value))
    assert "S3cret" not in rendered, rendered
    assert "ProgrammingError" in rendered


@pytest.mark.parametrize(
    "command",
    [
        ["db", "upgrade"],
        # Loads the local embedder before the first write.
        pytest.param(["ingest", "datasets/corpus"], marks=pytest.mark.slow),
    ],
)
def test_cli_output_carries_no_password(
    command: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VERITY_DATABASE_URL", _URL)
    get_settings.cache_clear()
    result = CliRunner().invoke(app, command)
    output = result.output + "".join(
        traceback.format_exception(result.exception) if result.exception else []
    )
    assert result.exit_code == 2, output
    assert "S3cret" not in output, output
