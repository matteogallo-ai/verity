"""Self-abstention: an answer whose text says the documents do not hold the answer.

The naive arm records no refusal, so a draft that itself declines would count as
answered. The patterns are the ones registered in
``docs/preregistration/naive-rag-baseline.md`` (metric 4) and reused by
``two-arm-comparison.md``; a test holds them equal to the registered text.
"""

from __future__ import annotations

import re

ABSTENTION_PATTERNS = (
    r"\b(do|does|did) not (contain|mention|provide|include|specify|say|state|address|cover|describe)\b",
    r"\b(don't|doesn't) (contain|mention|provide|include|specify|say|state|address|cover|describe)\b",
    r"\bno (information|mention|details?|data)\b",
    r"\b(is|are|was|were) not (mentioned|specified|stated|provided|covered|available|included|described)\b",
    r"\b(cannot|can't|unable to) (be )?(determine|answer|find|confirm)",
    r"\bI don't know\b",
)
_ABSTENTION_RE = [re.compile(p, re.IGNORECASE) for p in ABSTENTION_PATTERNS]


def is_self_abstention(text: str) -> bool:
    return any(p.search(text) for p in _ABSTENTION_RE)


__all__ = ["ABSTENTION_PATTERNS", "is_self_abstention"]
