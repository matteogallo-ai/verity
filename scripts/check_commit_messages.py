"""Fail when a commit subject in a range is not a Conventional Commit (rule 8).

Accepted subject: ``<type>[(<scope>)][!]: <description>`` with type one of
feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert; scope in
lowercase letters, digits and ``._/-``; a non-empty description. Merge commits
are skipped (GitHub creates them).

What it does NOT check: the body, the ~400-line size target, or whether the type
matches the change. Those stay with review.

Usage: ``python scripts/check_commit_messages.py <base>..<head>`` (CI passes the
pull request range). Exit 1 and one line per bad subject if any fails.
"""

from __future__ import annotations

import re
import subprocess
import sys

TYPES = (
    "feat",
    "fix",
    "docs",
    "style",
    "refactor",
    "perf",
    "test",
    "build",
    "ci",
    "chore",
    "revert",
)
SUBJECT_RE = re.compile(rf"^(?:{'|'.join(TYPES)})(?:\([a-z0-9._/-]+\))?!?: \S.*$")


def is_conventional(subject: str) -> bool:
    return bool(SUBJECT_RE.match(subject))


def subjects(rev_range: str) -> list[tuple[str, str]]:
    """``(short_sha, subject)`` for each non-merge commit in ``rev_range``."""
    out = subprocess.run(
        ["git", "log", "--no-merges", "--format=%h%x00%s", rev_range],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    pairs: list[tuple[str, str]] = []
    for line in out.splitlines():
        if line:
            sha, _, subject = line.partition("\x00")
            pairs.append((sha, subject))
    return pairs


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: check_commit_messages.py <base>..<head>", file=sys.stderr)
        return 2
    bad = [(sha, subject) for sha, subject in subjects(args[0]) if not is_conventional(subject)]
    for sha, subject in bad:
        print(f"{sha}: not a Conventional Commit: {subject}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
