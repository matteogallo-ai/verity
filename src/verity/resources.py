"""Locate the data files the package reads at run time: ``prompts/`` and ``migrations/``.

They live at the repository root, outside ``src/``. A source checkout (``uv sync``, an
editable install) reads them there. A wheel carries them inside the package as
``verity/_resources/<name>`` (``force-include`` in ``pyproject.toml``), so an installed
package does not depend on where the repository was. The installed copy wins, so a
wheel never reads a stray checkout.
"""

from __future__ import annotations

from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent


def resource_dir(name: str, *, package_dir: Path = _PACKAGE_DIR) -> Path:
    """The directory holding ``name`` (``"prompts"`` or ``"migrations"``).

    Raises ``FileNotFoundError`` naming both places looked at, instead of letting a
    later read fail on a path that never existed.
    """
    candidates = (package_dir / "_resources" / name, package_dir.parents[1] / name)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    looked = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"{name}/ not found; looked in {looked}")


__all__ = ["resource_dir"]
