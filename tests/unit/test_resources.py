"""prompts/ and migrations/ are found in a checkout and in an installed wheel."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from verity.resources import resource_dir


def test_a_checkout_reads_the_repository_directories() -> None:
    assert resource_dir("prompts") == Path("prompts").resolve()
    assert (resource_dir("migrations") / "001_initial.sql").is_file()


def test_the_packaged_copy_wins_over_a_checkout(tmp_path: Path) -> None:
    package = tmp_path / "site" / "lib" / "verity"
    (package / "_resources" / "prompts").mkdir(parents=True)
    (tmp_path / "site" / "prompts").mkdir()
    assert resource_dir("prompts", package_dir=package) == package / "_resources" / "prompts"


def test_a_missing_directory_names_where_it_looked(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="_resources/prompts"):
        resource_dir("prompts", package_dir=tmp_path / "a" / "b" / "verity")


@pytest.mark.slow
def test_the_wheel_ships_prompts_and_migrations(tmp_path: Path) -> None:
    import subprocess

    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path)], check=True, capture_output=True
    )
    [wheel] = tmp_path.glob("verity-*.whl")
    names = set(zipfile.ZipFile(wheel).namelist())
    assert "verity/_resources/prompts/synthesize.prompt" in names
    assert "verity/_resources/migrations/001_initial.sql" in names
