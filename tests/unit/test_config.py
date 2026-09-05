"""Settings load with sane defaults and honour environment overrides."""

from __future__ import annotations

import pytest

from verity.config import Settings, get_settings
from verity.types import Provider


def test_defaults_are_coherent() -> None:
    s = Settings()
    assert 0.0 <= s.confidence_threshold <= 1.0
    assert s.rerank_k <= s.retrieval_k  # can't keep more than we retrieve
    assert Provider.ANTHROPIC in s.provider_order
    assert s.embedding_dim > 0


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VERITY_CONFIDENCE_THRESHOLD", "0.8")
    monkeypatch.setenv("VERITY_RETRIEVAL_K", "50")
    s = Settings()
    assert s.confidence_threshold == 0.8
    assert s.retrieval_k == 50


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
