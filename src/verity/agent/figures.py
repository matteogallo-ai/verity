"""The figures of a text: the amounts, durations, dates and percentages a reader acts on.

Used by the ``figures`` support rule (verity.agent.support): a figure of the answer
must be in a verified quote, the question, or the cited section's heading.

Figures are compared as written values, not computed: "4'775.85", "4 775,85" and
"4775.85" are one figure, "31.00" and "31.-" are "31", but "2 200 h" is not derived
from "42.2 h x 52.14". A date such as "1.1.2026" is one figure. Ordinals keep their
digits ("13e" is 13). In a quote, number words count too ("trois ans" holds 3), so an
answer may write in digits what the text writes in letters; in the answer, only
digits are read.

Not figures, in the answer: list numbers ("1." at the start of a line) and the number
after a reference label ("art. 15 ch. 5", "al. 2", "annexe 3"). A wrong article number
is a sourcing error the scorer and the reader see, not a wrong amount.
"""

from __future__ import annotations

import re
import unicodedata
from decimal import Decimal
from typing import Literal

Language = Literal["en", "fr"]

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
# Swiss and French thousands separators: apostrophes, (narrow) no-break spaces, spaces.
_THOUSANDS_RE = re.compile("(?<=\\d)['\u2019\u02bc\u00a0\u202f ](?=\\d{3}(?!\\d))")
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*]\s*)?\d+[.)]\s", re.MULTILINE)
# A number right after one of these is a reference to a provision, not a figure.
_REFERENCE_BEFORE_RE = re.compile(
    r"(?:\bart(?:icles?|s)?\.?|\bal(?:in[eé]a)?s?\.?|\bch(?:iffres?)?\.?|\blet(?:tre)?\.?"
    r"|\bannexes?|\bsections?|\bclauses?|\bschedules?|\bn°|\bno\.?"
    r"|\b(?:et|and)|,)\s*$",
    re.IGNORECASE,
)
_REFERENCE_WINDOW = 14
_NUMBER_WORDS: dict[Language, dict[str, int]] = {
    "fr": {
        "deux": 2,
        "trois": 3,
        "quatre": 4,
        "cinq": 5,
        "six": 6,
        "sept": 7,
        "huit": 8,
        "neuf": 9,
        "dix": 10,
        "onze": 11,
        "douze": 12,
        "quinze": 15,
        "vingt": 20,
        "trente": 30,
        "quarante": 40,
        "cinquante": 50,
        "soixante": 60,
    },
    "en": {
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "fifteen": 15,
        "twenty": 20,
        "thirty": 30,
        "forty": 40,
        "fifty": 50,
        "sixty": 60,
    },
}


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


def _value(number: str) -> str:
    number = number.replace(",", ".").rstrip(".")
    if re.fullmatch(r"\d+(?:\.\d+)?", number):
        return format(Decimal(number).normalize(), "f")
    return number


def figures(text: str, *, language: Language, skip_references: bool = False) -> set[str]:
    """The figures of ``text`` as normalised values.

    ``skip_references`` reads ``text`` as an answer: list numbers and numbers after a
    reference label are left out, and number words are not read.
    """
    text = _THOUSANDS_RE.sub("", text)
    found: set[str] = set()
    if skip_references:
        text = _LIST_ITEM_RE.sub(" ", text)
    else:
        words = _NUMBER_WORDS[language]
        found |= {str(words[w]) for w in re.findall(r"[^\W\d_]+", _fold(text)) if w in words}
    for match in _NUMBER_RE.finditer(text):
        before = text[max(0, match.start() - _REFERENCE_WINDOW) : match.start()]
        if skip_references and _REFERENCE_BEFORE_RE.search(before):
            continue
        found.add(_value(match.group()))
    return found


__all__ = ["figures"]
