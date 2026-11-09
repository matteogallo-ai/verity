"""Verity: question answering over documents that cites the exact passage or refuses."""

from importlib.metadata import version

# pyproject.toml is the one place the version is written; an uninstalled source tree
# raises PackageNotFoundError here rather than report a stale number.
__version__ = version("verity")
