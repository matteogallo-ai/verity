"""The database the integration lane runs against: ``VERITY_DATABASE_URL`` (CI provisions
a pgvector service and sets it), else the settings default."""

from __future__ import annotations

import os

from verity.config import get_settings


def database_url() -> str:
    return os.environ.get("VERITY_DATABASE_URL") or get_settings().database_url.get_secret_value()
