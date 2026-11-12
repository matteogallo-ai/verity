"""VERITY_UI_MODE=presentation: the CCT demo shows a member what they would see.

/status tells the UI which mode and which corpus title to use; the web hides the
developer read-outs (web/tests/PresentationMode.test.tsx). The developer view stays
the default.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.unit.api_app import api_app
from tests.unit.test_demo_cct_profile import PROFILE, _sourced
from tests.unit.test_repo_rules import _serve_command
from verity.config import Settings


def test_the_developer_view_stays_the_default() -> None:
    settings = Settings()
    assert settings.ui_mode == "developer"
    assert settings.corpus_name is None
    with pytest.raises(ValidationError):
        Settings(ui_mode="kiosk")


def test_status_tells_the_ui_its_mode_and_corpus_name(tmp_path: Path) -> None:
    settings = Settings(
        api_auth_disabled=True, ui_mode="presentation", corpus_name="CCT Métal-Vaud"
    )
    with api_app(tmp_path, settings) as client:
        status = client.get("/status").json()
    assert status["ui_mode"] == "presentation"
    assert status["corpus_name"] == "CCT Métal-Vaud"


def test_status_reports_the_developer_view_by_default(tmp_path: Path) -> None:
    with api_app(tmp_path, Settings(api_auth_disabled=True)) as client:
        status = client.get("/status").json()
    assert status["ui_mode"] == "developer"
    assert status["corpus_name"] is None


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not installed")
def test_the_cct_demo_profile_presents() -> None:
    assert _sourced(PROFILE)["VERITY_UI_MODE"] == "presentation"


@pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")
def test_the_cct_demo_passes_the_corpus_name() -> None:
    command = _serve_command("demo-cct", "CORPUS=datasets/corpus", "NAME=Métal-Vaud")
    env, _, _ = command.partition("uv run verity serve")
    assert 'VERITY_CORPUS_NAME="Métal-Vaud"' in env


@pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")
def test_without_a_name_the_cct_demo_sets_none() -> None:
    command = _serve_command("demo-cct", "CORPUS=datasets/corpus")
    assert "VERITY_CORPUS_NAME" not in command
