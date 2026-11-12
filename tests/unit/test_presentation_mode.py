"""VERITY_UI_MODE=presentation: the CCT demo shows a member what they would see.

/status tells the UI which mode and which corpus title to use; the web hides the
developer read-outs (web/tests/PresentationMode.test.tsx). The developer view stays
the default.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.unit.api_app import api_app
from tests.unit.test_demo_cct_profile import PROFILE, _sourced
from verity.config import Settings

REPO_ROOT = Path(__file__).resolve().parents[2]


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


def _corpus_name_received(*variables: str) -> str:
    """Run the demo's command line up to the API start, and print the title it gets."""
    out = subprocess.run(
        ["make", "-n", "--no-print-directory", "demo-cct", "CORPUS=datasets/corpus", *variables],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    script = out[out.index("bash -c") :].replace(
        "uv run verity serve", "printenv VERITY_CORPUS_NAME; exit 0; uv run verity serve", 1
    )
    shown = subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent"},
    )
    return shown.stdout.strip()


needs_make = pytest.mark.skipif(
    shutil.which("make") is None or shutil.which("bash") is None,
    reason="make and bash are needed",
)


@needs_make
@pytest.mark.parametrize(
    "name", ["CCT Métal-Vaud", "CCT de l'industrie", 'L\'Union "X" $HOME, 100 %']
)
def test_the_cct_demo_passes_the_corpus_name_as_written(name: str) -> None:
    """French titles hold apostrophes; nothing in the name may break the command."""
    assert _corpus_name_received(f"CORPUS_NAME={name}") == name


@needs_make
def test_without_a_name_the_cct_demo_sets_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NAME", "a stray shell variable")
    assert _corpus_name_received() == ""
