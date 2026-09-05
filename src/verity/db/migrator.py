"""Tiny SQL-file migration runner.

Alembic would be overkill for the shape of this schema; a directory of numbered SQL
files and a ``schema_migrations`` table cover what we need (list applied, apply
pending, resistant to concurrent runs via a table-level advisory lock). Files are
executed in lexical order — the ``NNN_`` prefix on each file guarantees a total order.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import psycopg

# Resolve migrations/ relative to the *repo root* (two parents up from this file).
MIGRATIONS_DIR = (Path(__file__).resolve().parents[3] / "migrations").resolve()

# Postgres advisory-lock key — 63 bits, arbitrary but stable across restarts.
_ADVISORY_LOCK = 0x7E82BD5F51000001


@dataclass(frozen=True)
class MigrationFile:
    name: str
    path: Path
    sql: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.sql.encode("utf-8")).hexdigest()


def _load_files(directory: Path) -> list[MigrationFile]:
    if not directory.exists():
        return []
    files = sorted(p for p in directory.iterdir() if p.suffix == ".sql")
    return [MigrationFile(name=p.name, path=p, sql=p.read_text(encoding="utf-8")) for p in files]


def _ensure_bootstrap(cur: psycopg.Cursor[psycopg.rows.TupleRow]) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            name         TEXT PRIMARY KEY,
            sha256       TEXT NOT NULL,
            applied_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )


def list_applied(database_url: str) -> list[str]:
    """Return the names of migrations already recorded in ``schema_migrations``."""
    with psycopg.connect(database_url) as conn, conn.cursor() as cur:
        _ensure_bootstrap(cur)
        cur.execute("SELECT name FROM schema_migrations ORDER BY name")
        return [row[0] for row in cur.fetchall()]


def apply_pending(database_url: str, directory: Path | None = None) -> list[str]:
    """Apply every migration not yet recorded and return their names.

    Uses a Postgres advisory lock so concurrent runners can't race. Each migration
    is applied in its own transaction — a failing migration is rolled back cleanly
    and subsequent runs will re-attempt from the same point.
    """
    files = _load_files(directory or MIGRATIONS_DIR)
    applied: list[str] = []
    with psycopg.connect(database_url) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_lock(%s)", (_ADVISORY_LOCK,))
            try:
                _ensure_bootstrap(cur)
                cur.execute("SELECT name FROM schema_migrations")
                already = {row[0] for row in cur.fetchall()}
                for m in files:
                    if m.name in already:
                        continue
                    cur.execute("BEGIN")
                    try:
                        cur.execute(m.sql)
                        cur.execute(
                            "INSERT INTO schema_migrations (name, sha256) VALUES (%s, %s)",
                            (m.name, m.sha256),
                        )
                        cur.execute("COMMIT")
                    except Exception:
                        cur.execute("ROLLBACK")
                        raise
                    applied.append(m.name)
            finally:
                cur.execute("SELECT pg_advisory_unlock(%s)", (_ADVISORY_LOCK,))
    return applied


__all__ = ["MIGRATIONS_DIR", "MigrationFile", "apply_pending", "list_applied"]
