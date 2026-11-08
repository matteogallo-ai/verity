"""Each vulnerability osv-scanner.toml accepts keeps the reason it gives.

An exception says why the vulnerable code does not reach Verity. Where that reason is
"nothing calls these functions", the test below checks it over the installed packages,
so a dependency upgrade that starts calling them fails here instead of going unseen.
"""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Advisory id -> (package that holds the vulnerable code, functions no other code calls).
UNCALLED = {
    "GHSA-4j2p-28q2-5m79": (
        "accelerate",
        (b"load_checkpoint_in_model", b"load_checkpoint_and_dispatch"),
    ),
}


def _ignored_ids() -> set[str]:
    config = tomllib.loads((REPO_ROOT / "osv-scanner.toml").read_text())
    return {entry["id"] for entry in config.get("IgnoredVulns", [])}


def _callers(package: str, names: tuple[bytes, ...]) -> list[str]:
    spec = importlib.util.find_spec(package)
    assert spec is not None and spec.origin is not None, f"{package} is not installed"
    package_dir = Path(spec.origin).parent
    roots = [package_dir.parent, REPO_ROOT / "src", REPO_ROOT / "scripts"]
    found = []
    for root in roots:
        for path in root.rglob("*.py"):
            if package_dir in path.parents:
                continue
            text = path.read_bytes()
            if any(name in text for name in names):
                found.append(str(path))
    return found


def test_every_accepted_vulnerability_has_a_checked_reason() -> None:
    assert _ignored_ids() == set(UNCALLED)


def test_the_accepted_vulnerable_functions_are_never_called() -> None:
    for advisory, (package, names) in UNCALLED.items():
        assert _callers(package, names) == [], advisory
