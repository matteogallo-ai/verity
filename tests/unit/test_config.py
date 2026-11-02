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


def test_api_keys_never_appear_in_settings_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    """A logged or printed Settings object must not leak provider keys."""
    monkeypatch.setenv("VERITY_ANTHROPIC_API_KEY", "sk-ant-secret-canary")
    monkeypatch.setenv("VERITY_OPENAI_API_KEY", "sk-openai-secret-canary")
    s = Settings()
    for rendered in (repr(s), str(s), str(s.model_dump())):
        assert "secret-canary" not in rendered
    assert s.anthropic_api_key is not None
    assert s.anthropic_api_key.get_secret_value() == "sk-ant-secret-canary"


@pytest.mark.parametrize(
    ("anthropic", "openai", "expected"),
    [("", "", False), ("sk-a", "", True), ("", "sk-o", True), (None, None, False)],
)
def test_has_llm_api_key_treats_empty_values_as_unset(
    monkeypatch: pytest.MonkeyPatch, anthropic: str | None, openai: str | None, expected: bool
) -> None:
    for name, value in (("VERITY_ANTHROPIC_API_KEY", anthropic), ("VERITY_OPENAI_API_KEY", openai)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert Settings(_env_file=None).has_llm_api_key is expected


def test_database_url_password_never_appears_in_settings_repr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VERITY_DATABASE_URL", "postgresql://u:Hunter2Pw@h:5432/db")
    settings = Settings()
    assert "Hunter2Pw" not in repr(settings)
    assert "Hunter2Pw" not in str(settings.model_dump())
    assert settings.database_url.get_secret_value().endswith("@h:5432/db")
