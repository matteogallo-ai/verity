"""Tolerant extraction of the single JSON object an LLM was asked to return.

Real model outputs drift from "ONLY the JSON object": prose before or after it,
markdown code fences, or a one-element ``[{...}]`` wrapper. The synthesizer, the
confidence scorer and the faithfulness judge all need the same recovery, so it
lives here once. Callers decide what a ``None`` (unrecoverable output) means for
them: the scorer refuses, the synthesizer returns an empty draft that the agent
refuses (``no_citation``), the judge marks the example unsupported.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

# json.loads recurses in C once per nesting level. A long run of brackets raises
# RecursionError on the main thread but overflows the stack of a thread with a small
# one (512 KiB by default on macOS) and kills the process, so depth is checked before
# decoding. The prompts ask for two or three levels; brackets inside strings count too,
# which only makes the bound stricter.
_MAX_NESTING = 64


def load_json(text: str) -> Any:
    """``json.loads`` for model output: :class:`ValueError` (``JSONDecodeError`` is
    one) on invalid JSON or on nesting deeper than :data:`_MAX_NESTING`."""
    depth = 0
    for char in text:
        if char in "[{":
            depth += 1
            if depth > _MAX_NESTING:
                raise ValueError(f"JSON nested deeper than {_MAX_NESTING} levels")
        elif char in "]}":
            depth -= 1
    return json.loads(text)


def first_json_object(text: str) -> str:
    """Return the outermost ``{...}`` slice of ``text``.

    Falls back to the outermost ``[...]`` slice when there is no braced object
    (models that wrap the response in an array), and to ``""`` when neither exists.
    The slice runs from the first opening to the last closing bracket, so nested
    objects stay intact; two separate objects in one output yield invalid JSON,
    which callers treat as unparseable.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    array_start = text.find("[")
    array_end = text.rfind("]")
    if array_start != -1 and array_end != -1 and array_end > array_start:
        return text[array_start : array_end + 1]
    return ""


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse the JSON object in an LLM completion, or return ``None``.

    Accepts a bare object, an object wrapped in prose or code fences, and an array
    whose first dict element is taken as the object. Anything else (empty text,
    invalid JSON, a JSON scalar, an array without a dict) returns ``None``.
    """
    candidate = first_json_object(text) or text
    try:
        parsed = load_json(candidate)
    except ValueError:
        return None
    if isinstance(parsed, list):
        parsed = next((item for item in parsed if isinstance(item, dict)), None)
    return parsed if isinstance(parsed, dict) else None


def first_matching(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    """Value of the first key of ``keys`` present in ``mapping``, else ``None``.

    Lets parsers accept documented key aliases (``score`` / ``confidence``) without
    a chain of ``if key in`` checks.
    """
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


__all__ = ["first_json_object", "first_matching", "load_json", "parse_json_object"]
