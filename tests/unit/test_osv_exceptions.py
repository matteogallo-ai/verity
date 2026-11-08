"""Each vulnerability osv-scanner.toml accepts keeps the reason it gives.

Every exception there has the same kind of reason: nothing outside the vulnerable
package calls its vulnerable functions. The tests check that reason over the installed
packages and this repository, so a dependency upgrade that starts calling them fails
here instead of going unseen. An exception with another kind of reason needs its own
check before it is added.
"""

from __future__ import annotations

import ast
import importlib.util
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Advisory id -> (package that holds the vulnerable code, its vulnerable functions).
UNCALLED = {
    "GHSA-4j2p-28q2-5m79": (
        "accelerate",
        frozenset({"load_checkpoint_in_model", "load_checkpoint_and_dispatch"}),
    ),
}


def _ignored_ids() -> set[str]:
    config = tomllib.loads((REPO_ROOT / "osv-scanner.toml").read_text())
    return {entry["id"] for entry in config.get("IgnoredVulns", [])}


def _package_dir(package: str) -> Path:
    spec = importlib.util.find_spec(package)
    assert spec is not None and spec.origin is not None, f"{package} is not installed"
    return Path(spec.origin).parent


def _reaching(package_dir: Path, targets: frozenset[str]) -> set[str]:
    """Functions of the package that call a target, directly or through another one.

    Matching is by name, not by resolved import: it can only over-count, which makes
    the search outside the package stricter, never looser.
    """
    references: dict[str, set[str]] = {}
    for path in package_dir.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_bytes())):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                names = references.setdefault(node.name, set())
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Name):
                        names.add(inner.id)
                    elif isinstance(inner, ast.Attribute):
                        names.add(inner.attr)
    reaching = set(targets)
    while added := {f for f, refs in references.items() if refs & reaching} - reaching:
        reaching |= added
    return reaching


def _callers(package_dir: Path, names: set[str]) -> list[str]:
    encoded = [name.encode() for name in names]
    roots = [package_dir.parent, REPO_ROOT / "src", REPO_ROOT / "scripts"]
    return [
        str(path)
        for root in roots
        for path in root.rglob("*.py")
        if package_dir not in path.parents and any(n in path.read_bytes() for n in encoded)
    ]


def test_every_accepted_vulnerability_has_a_checked_reason() -> None:
    assert _ignored_ids() == set(UNCALLED)


def test_the_package_wrappers_of_the_vulnerable_functions_are_followed() -> None:
    """accelerate's own load_and_quantize_model calls load_checkpoint_in_model."""
    package, targets = UNCALLED["GHSA-4j2p-28q2-5m79"]
    assert "load_and_quantize_model" in _reaching(_package_dir(package), targets)


def test_the_accepted_vulnerable_functions_are_never_called() -> None:
    for advisory, (package, targets) in UNCALLED.items():
        package_dir = _package_dir(package)
        assert _callers(package_dir, _reaching(package_dir, targets)) == [], advisory
