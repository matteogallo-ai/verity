"""Annotate one judge-validation CSV in the terminal, one claim at a time.

    uv run python scripts/annotate.py datasets/eval/judge_validation/annotate_morning_1.csv
    uv run python scripts/annotate.py --review [--from N] <csv> [<csv> ...]

Keys (one keystroke, no Enter, either case): s supported · u unsupported · c comment ·
e show or hide the full extracts · b back · q quit (Ctrl-C quits too). Enter keeps the
current verdict and moves on. The CSV is rewritten after every change, and a new
session starts at the first claim without a verdict. ``--review`` starts at claim 1
(or N with ``--from N``) and walks every claim, answered or not, showing the current
verdict; s/u replace it, Enter keeps it.

Several CSVs given together form one session (claims numbered across files, e.g.
60/60); each answer is written back to the file its claim came from.

``--adjudicate <adjudication.csv>`` decides the claims of an adjudication file one by
one. That step is not blind: it shows the annotator's blind verdict and the judge's
next to the claim. s/u records the decision and asks for its reason, which is required
(c edits it); the blind labels live in other files and are never opened.

In annotation mode the tool is blind by construction: it reads the CSVs it is given and nothing else, and shows
only the claim, its cited passage, the extracts and what the annotator typed; neither
the item id nor any other column is displayed. It refuses a file that is not an
annotation CSV, so the answer key cannot be opened by mistake.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

LABELS = {"s": "supported", "u": "unsupported"}


@dataclass(frozen=True)
class Mode:
    verdict: str
    note: str
    required: tuple[str, ...]
    shown: tuple[tuple[str, str], ...] = ()
    note_required: bool = False


ANNOTATE = Mode("human_verdict", "comment", ("item_id", "claim", "cited_passage", "extracts"))
ADJUDICATE = Mode(
    "adjudicated_verdict",
    "reason",
    ("item_id", "claim", "cited_passage", "extracts", "blind_verdict", "judge_verdict"),
    shown=(("blind_verdict", "Your blind verdict"), ("judge_verdict", "Judge's verdict")),
    note_required=True,
)

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
class Sheet:
    """One annotation CSV: its rows are written back to it, in its own format."""

    path: Path
    fieldnames: list[str]
    rows: list[dict[str, str]]
    newline: str


@dataclass
class Session:
    sheets: list[Sheet]
    mode: Mode = ANNOTATE
    index: int = 0
    show_extracts: bool = False
    review: bool = False
    message: str = ""

    @property
    def rows(self) -> list[dict[str, str]]:
        return [row for sheet in self.sheets for row in sheet.rows]

    @property
    def done(self) -> int:
        return sum(1 for r in self.rows if r[self.mode.verdict])

    def sheet_at(self, index: int) -> Sheet:
        for sheet in self.sheets:
            if index < len(sheet.rows):
                return sheet
            index -= len(sheet.rows)
        raise IndexError(index)


def _read_sheet(path: Path, mode: Mode) -> Sheet:
    if path.suffix != ".csv":
        raise SystemExit(f"refusing to open {path.name}: give an annotation CSV")
    raw = path.read_bytes()
    newline = "\r\n" if b"\r\n" in raw else "\n"
    # utf-8-sig: a spreadsheet that re-saved the file may have added a BOM.
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    required = (*mode.required, mode.verdict)
    missing = [c for c in required if c not in fieldnames]
    if missing:
        raise SystemExit(f"{path.name} lacks column(s): {', '.join(missing)}")
    if mode.note not in fieldnames:
        fieldnames.append(mode.note)
    for line, row in enumerate(rows, start=2):
        if None in row or any(row[c] is None for c in required):
            raise SystemExit(f"{path.name} line {line}: wrong number of fields")
        row[mode.note] = row.get(mode.note) or ""
        verdict = row[mode.verdict].strip().lower()
        if verdict and verdict not in LABELS.values():
            raise SystemExit(
                f"{path.name} line {line}: {mode.verdict} {row[mode.verdict]!r} is not "
                f"one of {sorted(LABELS.values())}; fix or empty it"
            )
        row[mode.verdict] = verdict
    return Sheet(path, fieldnames, rows, newline)


def load(paths: Path | Sequence[Path], mode: Mode = ANNOTATE) -> Session:
    """One session over one or several CSVs, in the order given."""
    paths = [paths] if isinstance(paths, Path) else list(paths)
    if len({p.resolve() for p in paths}) != len(paths):
        raise SystemExit("the same CSV is given twice")
    session = Session([_read_sheet(p, mode) for p in paths], mode)
    rows = session.rows
    session.index = next((i for i, r in enumerate(rows) if not r[mode.verdict]), len(rows))
    return session


def save(sheet: Sheet) -> None:
    """Write one CSV to a temporary file then rename it: an interrupted save never
    leaves a half-written CSV. The file keeps its permissions and line endings."""
    fd, tmp = tempfile.mkstemp(dir=sheet.path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=sheet.fieldnames, lineterminator=sheet.newline)
            writer.writeheader()
            writer.writerows(sheet.rows)
        shutil.copymode(sheet.path, tmp)
        os.replace(tmp, sheet.path)
    except BaseException:
        # Re-raised: only the temporary file is cleaned up, the error is the caller's.
        Path(tmp).unlink(missing_ok=True)
        raise


def _counted(mode: Mode) -> str:
    return "adjudicated" if mode is ADJUDICATE else "annotated"


def render(session: Session) -> str:
    mode = session.mode
    total = len(session.rows)
    done = _counted(mode)
    if session.index >= total:
        return f"{CLEAR}{session.done}/{total} {done}. b back · q quit\n"
    row = session.rows[session.index]
    labels = ["ADJUDICATE"] * (mode is ADJUDICATE) + ["REVIEW"] * session.review
    header = f"{' '.join(labels)}   " if labels else ""
    lines = [
        CLEAR,
        f"{header}{session.index + 1}/{total}   ({session.done} {done})\n\n",
        f"{BOLD}{sanitize(row['claim'])}{RESET}\n\n",
        f"Cited passage:\n{highlight(row['cited_passage'])}\n\n",
    ]
    if session.show_extracts:
        lines.append(f"Extracts:\n{highlight(row['extracts'])}\n\n")
    lines.extend(f"{label}: {sanitize(row[column])}\n" for column, label in mode.shown)
    if row[mode.verdict]:
        lines.append(f"Current verdict: {row[mode.verdict]} (Enter keeps it)\n")
    elif session.review or mode is ADJUDICATE:
        lines.append("Current verdict: none\n")
    if row[mode.note]:
        lines.append(f"{mode.note.capitalize()}: {sanitize(row[mode.note])}\n")
    if session.message:
        lines.append(f"\n{session.message}\n")
    lines.append(f"\ns supported · u unsupported · c {mode.note} · e extracts · b back · q quit\n")
    return "".join(lines)


def step(session: Session, key: str, ask: Callable[[str], str]) -> bool:
    """Apply one keystroke. Returns False when the user quits."""
    # Shift or Caps Lock must not disable a command; escape sequences, whose final
    # letter could otherwise act as one, never reach this point (see keys_from).
    key = key.lower()
    session.message = ""
    mode = session.mode
    if key == "q":
        return False
    if key == "b":
        session.index = max(0, session.index - 1)
        session.show_extracts = False
        return True
    if session.index >= len(session.rows):
        return True
    row = session.rows[session.index]
    sheet = session.sheet_at(session.index)
    if key in LABELS:
        if mode.note_required:
            # A reason written for the other verdict would contradict the new one.
            kept = row[mode.note] if row[mode.verdict] == LABELS[key] else ""
            keep = " (Enter keeps it)" if kept else ""
            typed = ask(f"Reason for {LABELS[key]}{keep}: ").strip()
            if typed == "-":
                typed = ""
            if not (typed or kept):
                session.message = "A reason is required: nothing recorded."
                return True
            row[mode.note] = typed or kept
        row[mode.verdict] = LABELS[key]
        save(sheet)
        session.index += 1
        session.show_extracts = False
    elif key == "c":
        clear = "" if mode.note_required else ", '-' to clear"
        typed = ask(f"{mode.note.capitalize()} (Enter to keep{clear}): ").strip()
        if typed == "-":
            # A required reason cannot be emptied; it can only be replaced.
            row[mode.note] = row[mode.note] if mode.note_required else ""
        elif typed:
            row[mode.note] = typed
        save(sheet)
    elif key == "e":
        session.show_extracts = not session.show_extracts
    elif key in ("\r", "\n") and row[mode.verdict]:
        # Keeps an existing verdict; an unanswered claim is never skipped silently.
        session.index += 1
        session.show_extracts = False
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="annotate.py", description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("csv", type=Path, nargs="+", help="one or more CSVs, one session")
    parser.add_argument("--review", action="store_true", help="walk every claim from claim 1")
    parser.add_argument("--from", dest="start", type=int, metavar="N", help="start at claim N")
    parser.add_argument(
        "--adjudicate",
        action="store_true",
        help="decide an adjudication CSV (shows the blind and judge verdicts; reason required)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    except SystemExit as err:
        return int(err.code or 0)
    session = load(args.csv, ADJUDICATE if args.adjudicate else ANNOTATE)
    if args.review:
        session.review = True
        session.index = 0
    if args.start is not None:
        if not 1 <= args.start <= len(session.rows):
            print(f"--from must be between 1 and {len(session.rows)}", file=sys.stderr)
            return 2
        session.index = args.start - 1

    def out(text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    # Ctrl-C is a way to quit: every answer is already on disk.
    with contextlib.suppress(KeyboardInterrupt):
        run(session, _terminal_keys(), _ask, out)
    names = ", ".join(str(sheet.path) for sheet in session.sheets)
    out(f"\nSaved {names} ({session.done}/{len(session.rows)} {_counted(session.mode)}).\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
