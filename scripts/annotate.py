"""Annotate one judge-validation CSV in the terminal, one claim at a time.

    uv run python scripts/annotate.py datasets/eval/judge_validation/annotate_morning_1.csv

Keys (one keystroke, no Enter): s supported · u unsupported · c comment · e show or
hide the full extracts · b back · q quit. The CSV is rewritten after every answer,
and a new session starts at the first claim without a verdict.

Blind by construction: the tool reads the CSV it is given and nothing else. It never
opens the answer key or the judge's verdicts, refuses to run on them, and shows only
the claim, its cited passage and the extracts; the opaque item id is not shown either.
"""

from __future__ import annotations

import csv
import os
import re
import sys
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

LABELS = {"s": "supported", "u": "unsupported"}
REQUIRED = ["item_id", "claim", "cited_passage", "extracts", "human_verdict"]
COMMENT = "comment"
# Files that hold the answer to the blind question: reading them would unblind the run.
FORBIDDEN_NAMES = {"judge_key.json", "perturbed_verdicts.json", "agreement.json"}

BOLD, YELLOW, RESET = "\033[1m", "\033[33;1m", "\033[0m"
CLEAR = "\033[2J\033[H"

_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
# Numbers (incl. money, percentages, decimals), month names, negations, and capitalised
# words that are not the first word of a sentence: the spans a perturbation changes.
_HIGHLIGHT = re.compile(
    rf"(?:US)?\$?\d[\d,]*(?:\.\d+)?%?"
    rf"|\b(?:{_MONTHS})\b"
    r"|\b(?:not|no|never|none|nor|cannot|without)\b|\b\w+n't\b"
    r"|(?<=[a-z,;:] )[A-Z][\w&-]*(?: [A-Z][\w&-]*)*"
)


def highlight(text: str) -> str:
    return _HIGHLIGHT.sub(lambda m: f"{YELLOW}{m.group()}{RESET}", text)


@dataclass
class Session:
    path: Path
    fieldnames: list[str]
    rows: list[dict[str, str]]
    index: int = 0
    show_extracts: bool = False
    history: list[int] = field(default_factory=list)

    @property
    def done(self) -> int:
        return sum(1 for r in self.rows if r["human_verdict"].strip())


def load(path: Path) -> Session:
    if path.name in FORBIDDEN_NAMES or path.suffix != ".csv":
        raise SystemExit(f"refusing to open {path.name}: give an annotation CSV")
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    missing = [c for c in REQUIRED if c not in fieldnames]
    if missing:
        raise SystemExit(f"{path.name} lacks column(s): {', '.join(missing)}")
    if COMMENT not in fieldnames:
        fieldnames.append(COMMENT)
    for row in rows:
        row.setdefault(COMMENT, "")
        row[COMMENT] = row[COMMENT] or ""
    first_open = next((i for i, r in enumerate(rows) if not r["human_verdict"].strip()), None)
    return Session(path, fieldnames, rows, len(rows) if first_open is None else first_open)


def save(session: Session) -> None:
    """Write to a temporary file then rename it: an interrupted save never leaves a
    half-written CSV."""
    fd, tmp = tempfile.mkstemp(dir=session.path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=session.fieldnames)
            writer.writeheader()
            writer.writerows(session.rows)
        os.replace(tmp, session.path)
    except BaseException:
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
        f"{BOLD}{row['claim']}{RESET}\n\n",
        f"Cited passage:\n{highlight(row['cited_passage'])}\n\n",
    ]
    if session.show_extracts:
        lines.append(f"Extracts:\n{highlight(row['extracts'])}\n\n")
    if row["human_verdict"].strip():
        lines.append(f"Current verdict: {row['human_verdict']}\n")
    if row[COMMENT]:
        lines.append(f"Comment: {row[COMMENT]}\n")
    lines.append("\ns supported · u unsupported · c comment · e extracts · b back · q quit\n")
    return "".join(lines)


def step(session: Session, key: str, ask: Callable[[str], str]) -> bool:
    """Apply one keystroke. Returns False when the user quits."""
    if key == "q":
        return False
    if key == "b":
        if session.history:
            session.index = session.history.pop()
        elif session.index > 0:
            session.index -= 1
        session.show_extracts = False
        return True
    if session.index >= len(session.rows):
        return True
    row = session.rows[session.index]
    if key in LABELS:
        row["human_verdict"] = LABELS[key]
        save(session)
        session.history.append(session.index)
        session.index += 1
        session.show_extracts = False
    elif key == "c":
        row[COMMENT] = ask("Comment (Enter to keep, '-' to clear): ").strip() or row[COMMENT]
        if row[COMMENT] == "-":
            row[COMMENT] = ""
        save(session)
    elif key == "e":
        session.show_extracts = not session.show_extracts
    return True


def run(
    session: Session, keys: Iterable[str], ask: Callable[[str], str], out: Callable[[str], None]
) -> None:
    out(render(session))
    for key in keys:
        if not step(session, key.lower(), ask):
            break
        out(render(session))


def _keystrokes() -> Iterable[str]:
    import termios
    import tty

    fd = sys.stdin.fileno()
    while True:
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        if key in ("", "\x03", "\x04"):  # EOF, Ctrl-C, Ctrl-D
            return
        yield key


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    session = load(Path(args[0]))

    def out(text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    run(session, _keystrokes(), input, out)
    out(f"\nSaved {session.path} ({session.done}/{len(session.rows)} annotated).\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
