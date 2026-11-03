"""Annotate one judge-validation CSV in the terminal, one claim at a time.

    uv run python scripts/annotate.py datasets/eval/judge_validation/annotate_morning_1.csv

Keys (one keystroke, no Enter): s supported · u unsupported · c comment · e show or
hide the full extracts · b back · q quit (Ctrl-C quits too). The CSV is rewritten
after every answer, and a new session starts at the first claim without a verdict.

Blind by construction: the tool reads the CSV it is given and nothing else, and shows
only the claim, its cited passage, the extracts and what the annotator typed; neither
the item id nor any other column is displayed. It refuses a file that is not an
annotation CSV, so the answer key cannot be opened by mistake.
"""

from __future__ import annotations

import contextlib
import csv
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

LABELS = {"s": "supported", "u": "unsupported"}
REQUIRED = ["item_id", "claim", "cited_passage", "extracts", "human_verdict"]
COMMENT = "comment"

BOLD, YELLOW, RESET = "\033[1m", "\033[33;1m", "\033[0m"
CLEAR = "\033[2J\033[H"
ESC = "\x1b"

_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
# Numbers (incl. money, percentages, decimals), month names, negations, and capitalised
# words that are not the first word of a sentence: the spans a perturbation changes.
_HIGHLIGHT = re.compile(
    r"-?(?:US)?\$?\d+(?:,\d{3})*(?:\.\d+)?%?"
    rf"|\b(?:{_MONTHS})\b"
    r"|\b(?:not|no|never|none|nor|cannot|without)\b|\b\w+n['\u2019]t\b"
    r"|(?<=[a-z,;:] )[A-Z][\w&-]*(?: [A-Z][\w&-]*)*"
)
# Document text is untrusted: an escape sequence inside it could clear or rewrite the
# screen and change what the annotator reads. Tabs and newlines are kept.
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def sanitize(text: str) -> str:
    return _CONTROL.sub("", text)


def highlight(text: str) -> str:
    return _HIGHLIGHT.sub(lambda m: f"{YELLOW}{m.group()}{RESET}", sanitize(text))


@dataclass
class Session:
    path: Path
    fieldnames: list[str]
    rows: list[dict[str, str]]
    newline: str
    index: int = 0
    show_extracts: bool = False

    @property
    def done(self) -> int:
        return sum(1 for r in self.rows if r["human_verdict"])


def load(path: Path) -> Session:
    if path.suffix != ".csv":
        raise SystemExit(f"refusing to open {path.name}: give an annotation CSV")
    raw = path.read_bytes()
    newline = "\r\n" if b"\r\n" in raw else "\n"
    # utf-8-sig: a spreadsheet that re-saved the file may have added a BOM.
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    missing = [c for c in REQUIRED if c not in fieldnames]
    if missing:
        raise SystemExit(f"{path.name} lacks column(s): {', '.join(missing)}")
    if COMMENT not in fieldnames:
        fieldnames.append(COMMENT)
    for line, row in enumerate(rows, start=2):
        if None in row or any(row[c] is None for c in REQUIRED):
            raise SystemExit(f"{path.name} line {line}: wrong number of fields")
        row[COMMENT] = row.get(COMMENT) or ""
        verdict = row["human_verdict"].strip().lower()
        if verdict and verdict not in LABELS.values():
            raise SystemExit(
                f"{path.name} line {line}: human_verdict {row['human_verdict']!r} is not "
                f"one of {sorted(LABELS.values())}; fix or empty it"
            )
        row["human_verdict"] = verdict
    first_open = next((i for i, r in enumerate(rows) if not r["human_verdict"]), len(rows))
    return Session(path, fieldnames, rows, newline, first_open)


def save(session: Session) -> None:
    """Write to a temporary file then rename it: an interrupted save never leaves a
    half-written CSV. The file keeps its permissions and line endings."""
    fd, tmp = tempfile.mkstemp(dir=session.path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=session.fieldnames, lineterminator=session.newline
            )
            writer.writeheader()
            writer.writerows(session.rows)
        shutil.copymode(session.path, tmp)
        os.replace(tmp, session.path)
    except BaseException:
        # Re-raised: only the temporary file is cleaned up, the error is the caller's.
        Path(tmp).unlink(missing_ok=True)
        raise


def render(session: Session) -> str:
    total = len(session.rows)
    if session.index >= total:
        return f"{CLEAR}{session.done}/{total} annotated. b back · q quit\n"
    row = session.rows[session.index]
    lines = [
        CLEAR,
        f"{session.index + 1}/{total}   ({session.done} annotated)\n\n",
        f"{BOLD}{sanitize(row['claim'])}{RESET}\n\n",
        f"Cited passage:\n{highlight(row['cited_passage'])}\n\n",
    ]
    if session.show_extracts:
        lines.append(f"Extracts:\n{highlight(row['extracts'])}\n\n")
    if row["human_verdict"]:
        lines.append(f"Current verdict: {row['human_verdict']}\n")
    if row[COMMENT]:
        lines.append(f"Comment: {sanitize(row[COMMENT])}\n")
    lines.append("\ns supported · u unsupported · c comment · e extracts · b back · q quit\n")
    return "".join(lines)


def step(session: Session, key: str, ask: Callable[[str], str]) -> bool:
    """Apply one keystroke. Returns False when the user quits."""
    if key == "q":
        return False
    if key == "b":
        session.index = max(0, session.index - 1)
        session.show_extracts = False
        return True
    if session.index >= len(session.rows):
        return True
    row = session.rows[session.index]
    if key in LABELS:
        row["human_verdict"] = LABELS[key]
        save(session)
        session.index += 1
        session.show_extracts = False
    elif key == "c":
        typed = ask("Comment (Enter to keep, '-' to clear): ").strip()
        row[COMMENT] = "" if typed == "-" else (typed or row[COMMENT])
        save(session)
    elif key == "e":
        session.show_extracts = not session.show_extracts
    return True


def run(
    session: Session, keys: Iterable[str], ask: Callable[[str], str], out: Callable[[str], None]
) -> None:
    out(render(session))
    for key in keys:
        if not step(session, key, ask):
            break
        out(render(session))


def keys_from(read: Callable[[], str]) -> Iterable[str]:
    """Single keystrokes from ``read``. An escape sequence (arrow keys send ESC [ B)
    is swallowed whole, so its trailing letter never acts as a command."""
    while True:
        key = read()
        if key in ("", "\x04"):  # EOF, Ctrl-D
            return
        if key == ESC:
            nxt = read()
            if nxt in ("[", "O"):
                while (ch := read()) and not (ch.isalpha() or ch == "~"):
                    pass
            continue
        yield key


def _terminal_keys() -> Iterable[str]:
    import termios
    import tty

    fd = sys.stdin.fileno()

    def read() -> str:
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    return keys_from(read)


def _ask(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        return ""


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    session = load(Path(args[0]))

    def out(text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    # Ctrl-C is a way to quit: every answer is already on disk.
    with contextlib.suppress(KeyboardInterrupt):
        run(session, _terminal_keys(), _ask, out)
    out(f"\nSaved {session.path} ({session.done}/{len(session.rows)} annotated).\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
