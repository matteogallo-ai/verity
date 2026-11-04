"""Deterministic perturbations that turn a supported claim into a false but plausible one.

A judge that labels every claim "supported" agrees with an annotator who sees almost
only supported claims, and Cohen's κ cannot tell it from a judge that reads the
evidence. Claims made false on purpose measure what κ cannot: how many false claims
the judge catches. Each perturbation changes one token span, so the claim stays
fluent and on topic:

- ``number``: one figure (not a year) scaled to a value absent from the evidence;
- ``date``: a month name shifted, else a year moved to one the evidence does not hold;
- ``negation``: ``not`` inserted after the first auxiliary (``can`` → ``cannot``);
- ``entity``: a name replaced by another name of the same type (person, city,
  document...), preferably one the evidence also holds.

A claim with nothing to change for a kind gets ``None``, never a forced edit. The
result is false with respect to the evidence in the common case, not by proof (the
evidence can hold the new value in words, or say both things): the human annotator
labels perturbed claims blind, like the others, and that label is the reference.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum


class Kind(StrEnum):
    NUMBER = "number"
    DATE = "date"
    NEGATION = "negation"
    ENTITY = "entity"


@dataclass(frozen=True)
class Perturbation:
    kind: Kind
    original: str
    perturbed: str
    old: str
    new: str


MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)

# A figure not glued to letters (FY2024, SOC2 are identifiers), with optional
# thousands separators and decimals.
_NUMBER_RE = re.compile(r"(?<![\w.,-])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w-])")
# A figure after these words is a reference, not a quantity ("Section 4.2").
_REFERENCE_RE = re.compile(r"(?:Section|Article|Clause|Schedule|Exhibit|Annex|No\.|§)\s*$")
_YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_MONTH_RE = re.compile(r"\b(" + "|".join(MONTHS) + r")\b")
_NEGATIVE_RE = re.compile(r"\b(?:not|no|never|none|cannot|nor)\b|n't", re.IGNORECASE)
_AUX_RE = re.compile(
    r"(?<![\w-])(is|are|was|were|has|have|had|does|do|did|will|would|should|must|may|might|could|can)(?![\w-])"
)
# "has" / "have" / "had" are auxiliaries only before a participle; "has three tiers"
# would become "has not three tiers".
_PERFECT_AUX = {"has", "have", "had"}
# "-ed" participles, "been", and the common "-en" ones: "fifteen" or "children" end in
# "en" too, and "has not fifteen days" is not English.
_PARTICIPLE_RE = re.compile(
    r"\s+(?:been|\w+ed|taken|given|written|chosen|broken|driven|forgotten|seen|shown|known|"
    r"grown|drawn|proven|spoken|stolen|frozen|hidden|ridden|risen|beaten|eaten|fallen)\b"
)
_SCALES = (1.5, 0.5, 2.0, 1.25, 3.0)
_MONTH_SHIFTS = (3, 6, 9, 1)
_YEAR_SHIFTS = (1, -1, 2, -2)


def perturb(
    claim: str,
    kind: Kind,
    evidence: str,
    entities: Mapping[str, Sequence[str]] | None = None,
) -> Perturbation | None:
    """One false-but-plausible edit of ``claim`` of the given kind, or ``None``.

    ``entities`` maps a type (person, city, document...) to the names of that type;
    an entity is swapped only for a name of the same type, so a person never becomes
    a company. Without it the entity kind has nothing to swap."""
    if kind is Kind.ENTITY:
        edit = _edit_entity(claim, evidence, entities or {})
    else:
        edit = _EDITORS[kind](claim, evidence)
    if edit is None:
        return None
    start, end, new = edit
    return Perturbation(
        kind=kind,
        original=claim,
        perturbed=claim[:start] + new + claim[end:],
        old=claim[start:end],
        new=new,
    )


def _is_year(token: str) -> bool:
    return token.isdigit() and len(token) == 4 and 1900 <= int(token) <= 2099


def _format_like(value: float, original: str) -> str:
    decimals = len(original.split(".")[1]) if "." in original else 0
    return f"{value:,.{decimals}f}" if "," in original else f"{value:.{decimals}f}"


def _next_to_month(text: str, start: int, end: int) -> bool:
    before = re.findall(r"\w+", text[:start])
    after = re.findall(r"\w+", text[end:])
    return bool((before and before[-1] in MONTHS) or (after and after[0] in MONTHS))


def _edit_number(claim: str, evidence: str) -> tuple[int, int, str] | None:
    for match in _NUMBER_RE.finditer(claim):
        token = match.group()
        if _is_year(token):
            continue
        # "three (3) years" restates a number word, "401(k)" names a plan, "March 31"
        # is a date: changing them is not a plausible number error.
        after = claim[match.end() :]
        if after.startswith("(") or (
            claim[match.start() - 1 : match.start()] == "(" and after.startswith(")")
        ):
            continue
        if _next_to_month(claim, match.start(), match.end()):
            continue
        if _REFERENCE_RE.search(claim[: match.start()]):
            continue
        value = float(token.replace(",", ""))
        percent = claim[match.end() :].lstrip().startswith(("%", "percent"))
        for scale in _SCALES:
            scaled = value * scale if value else 5.0
            new = _format_like(scaled, token)
            if percent and value <= 100 < scaled:
                continue
            if new != token and new not in evidence:
                return match.start(), match.end(), new
    return None


def _edit_date(claim: str, evidence: str) -> tuple[int, int, str] | None:
    words = set(re.findall(r"\w+", evidence))
    month = _MONTH_RE.search(claim)
    if month is not None:
        index = MONTHS.index(month.group())
        for shift in _MONTH_SHIFTS:
            new = MONTHS[(index + shift) % 12]
            if new not in words:
                return month.start(), month.end(), new
    for year in _YEAR_RE.finditer(claim):
        for shift in _YEAR_SHIFTS:
            new = str(int(year.group()) + shift)
            if new not in evidence:
                return year.start(), year.end(), new
    return None


def _edit_negation(claim: str, evidence: str) -> tuple[int, int, str] | None:
    del evidence  # the edit is grammatical; the evidence does not choose it
    if _NEGATIVE_RE.search(claim):
        return None
    for match in _AUX_RE.finditer(claim):
        aux = match.group()
        if aux in _PERFECT_AUX and not _PARTICIPLE_RE.match(claim, match.end()):
            continue
        if re.search(r"\bto\s+$", claim[: match.start()]):
            continue  # "need to do": a main verb after "to", not an auxiliary
        return match.start(), match.end(), "cannot" if aux == "can" else f"{aux} not"
    return None


_OPENERS = frozenset(
    ["A", "An", "Any", "Each", "For", "If", "In", "Its", "The", "This", "Under", "When", "Where"]
)


def _inside_longer_name(text: str, start: int, end: int) -> bool:
    """ "Northwind" in "Northwind Logistics": a capitalised word right before or after
    makes it part of another name."""
    before = re.search(r"([A-Z][\w-]*) $", text[:start])
    # "Director of Finance" is a title of its own.
    after = re.match(r" (?:of )?[A-Z]", text[end:])
    return bool(after or (before and before.group(1) not in _OPENERS))


def _edit_entity(
    claim: str, evidence: str, entities: Mapping[str, Sequence[str]]
) -> tuple[int, int, str] | None:
    # Longest names first: "Enterprise Cloud" is a segment, not the edition "Enterprise".
    typed = sorted(
        ((name, kind) for kind, names in entities.items() for name in names),
        key=lambda item: (-len(item[0]), item[0]),
    )
    for name, kind in typed:
        match = next(
            (
                m
                for m in re.finditer(rf"(?<![\w-]){re.escape(name)}(?![\w-])", claim)
                if not _inside_longer_name(claim, m.start(), m.end())
            ),
            None,
        )
        if match is None:
            continue
        others = [n for n in entities[kind] if n != name and n not in claim]
        if not others:
            continue
        # A name the evidence also holds is the distractor a real error would pick.
        new = min(others, key=lambda n: (n not in evidence, n))
        return match.start(), match.end(), new
    return None


_EDITORS = {
    Kind.NUMBER: _edit_number,
    Kind.DATE: _edit_date,
    Kind.NEGATION: _edit_negation,
}
