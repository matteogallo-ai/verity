"""Fail when shipped code or docs talk about sprints, the future, or untracked TODOs.

Rule 2 of docs/ENGINEERING_STANDARDS.md: shipped code says what the code does
now. Sprint labels ("S4.2"), promises ("lands in S3", "coming in v1.2") and
TODOs without an issue link go stale the moment they are merged.

Scanned: src/, docs/, prompts/, README.md, web/README.md, datasets/eval/README.md
(text files only). Excluded: docs/ENGINEERING_STANDARDS.md (quotes the patterns as
examples) and CHANGELOG.md (history is its job).

What it checks, per line:
- ``sprint``: a sprint label S0-S7, optionally with a sub-number (S4.2), a range
  (S4-S7, S1-S4) or a plus (S3+);
- ``future``: "lands / landing / landed / wired in|at S<n>|v<n>",
  "coming / planned / scheduled in|for S<n>|v<n>", "Implementation(s) coming";
- ``todo``: TODO / FIXME / XXX not followed by an issue link "(#<number>)".

What it does NOT check: future tense phrased any other way ("will be added",
"later"), version numbers hard-coded outside package metadata, or whether a
statement is true. Those stay with human review (rule 10).

Usage: ``python scripts/check_stale_narrative.py [--root PATH]``.
Output: ``file:line: type: excerpt`` per finding; exit 1 if any, 0 otherwise.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SCANNED_DIRS = ("src", "docs", "prompts")
SCANNED_FILES = ("README.md", "web/README.md", "datasets/eval/README.md")
EXCLUDED = frozenset({"docs/ENGINEERING_STANDARDS.md", "CHANGELOG.md"})
TEXT_SUFFIXES = frozenset({".py", ".md", ".prompt", ".sql", ".txt", ".toml", ".yml", ".yaml"})

_SPRINT = r"S[0-7](?:\.\d+)?"
PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "sprint",
        # Not preceded by a word character, a path separator, a dot or
        # "Amazon "/"AWS " (the storage service); a hyphen may precede ("pre-S6.1");
        # ranges accept a hyphen or an en dash (\u2013).
        re.compile(
            rf"(?<![\w/.])(?<!Amazon )(?<!AWS ){_SPRINT}"
            rf"(?:\s*[-\u2013]\s*{_SPRINT}|\+)?(?![\w-])"
        ),
    ),
    (
        "future",
        re.compile(
            r"\b(?:lands?|landing|landed|wired)\s+(?:in|at)\s+(?:S\d|v\d)"
            r"|\b(?:coming|planned|scheduled)\s+(?:in|for)\s+(?:S\d|v\d)"
            r"|\bImplementations?\s+coming\b",
            re.IGNORECASE,
        ),
    ),
    ("todo", re.compile(r"\b(?:TODO|FIXME|XXX)\b(?!\s*\(#\d+\))")),
)


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    kind: str
    excerpt: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.kind}: {self.excerpt}"


def iter_scanned_files(root: Path) -> Iterator[Path]:
    """Yield every scanned text file under ``root``, in a stable order."""
    candidates: list[Path] = []
    for directory in SCANNED_DIRS:
        base = root / directory
        if base.is_dir():
            candidates.extend(p for p in base.rglob("*") if p.is_file())
    candidates.extend(root / f for f in SCANNED_FILES if (root / f).is_file())
    for path in sorted(set(candidates)):
        rel = path.relative_to(root).as_posix()
        if rel in EXCLUDED or path.suffix not in TEXT_SUFFIXES:
            continue
        if any(part in {"__pycache__", "node_modules"} for part in path.parts):
            continue
        yield path


def scan_text(rel_path: str, text: str) -> list[Finding]:
    """Return one finding per (line, kind) match in ``text``."""
    findings: list[Finding] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for kind, pattern in PATTERNS:
            if pattern.search(line):
                findings.append(Finding(rel_path, lineno, kind, line.strip()[:160]))
    return findings


def scan(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for path in iter_scanned_files(root):
        rel = path.relative_to(root).as_posix()
        findings.extend(scan_text(rel, path.read_text(encoding="utf-8")))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="Repository root.")
    args = parser.parse_args(argv)
    findings = scan(args.root)
    for finding in findings:
        print(finding)
    print(f"{len(findings)} stale-narrative finding(s)", file=sys.stderr)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
