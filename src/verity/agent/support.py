"""Deterministic support check on a cited draft, run before the confidence scorer.

The scorer is a model reading untrusted documents, so a document can try to dictate
its verdict. Its score therefore never ships an answer on its own: the draft must
also pass two checks computed from data the documents cannot rewrite.

- **Citation coverage**: the share of the answer's content words that appear in its
  cited quotes. An answer that says things its quotes do not is refused
  (``low_citation_coverage``). Words are compared on their first five letters, so
  ``terminate`` matches ``termination``; numbers are compared whole.
- **Cited relevance**: the best cross-encoder score among the cited chunks. A draft
  whose citations the reranker judged irrelevant to the question is refused
  (``irrelevant_citation``). Skipped when no cited hit carries a reranker score
  (reranker disabled), because other stages' scores are on other scales.

- **Injection markers**: a draft citing a document in which any retrieved chunk
  carries a known injection pattern (this system's verdict keys as JSON, "ignore
  previous instructions", a ``SYSTEM:`` role line, a neutralised delimiter tag) is
  refused (``suspected_injection``). The whole document is distrusted, so an
  instruction in one section cannot vouch for a false fact in another.

What this does not stop: an injection paraphrased past the marker list, or a poisoned
document that is relevant to the question and states a false answer verbatim. That is
corpus poisoning; the defence is control over what is ingested (see SECURITY.md).

What this gets wrong by design:
- Coverage compares words, not meaning: "may not terminate" is fully covered by a
  quote saying "may terminate". A lexical polarity rule, measured on the live runs,
  refused correct answers ("No, ... is explicitly prohibited"), so polarity is left to
  the confidence scorer (docs/audit/2026-11-03-neutral-audit.md).
- Coverage is lexical, so it needs the corpus language: ``en`` (the default) splits on
  ASCII letters and digits; ``fr`` splits on Unicode letters, compares accent-folded
  words and drops French stopwords, since the English rule cuts "réalisé" into "alis"
  and counts "dans" or "pour" as content. The threshold is the same for both.
- Coverage does not check figures: "60 days" is covered by a quote saying "30 days"
  if the other words match. A rule requiring every number and date to be quoted,
  measured on the live runs, refused correct answers that compute or restate a figure
  (docs/audit/2026-11-03-figure-rule-measurement.md), so it is not applied.
- The marker list is fixed patterns: a benign document with a ``System:`` line or a
  ``"score":`` key is distrusted as a whole, and every answer citing it is refused.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

from verity.agent.evidence import DELIMITER_TAGS, neutralise
from verity.types import Citation, RefusalReason, RetrievalHit, RetrieverKind

Language = Literal["en", "fr"]

_WORD_RE = re.compile(r"[a-z0-9]+")
# Letters and digits of any script; ``_`` is a word character but not a letter.
_UNICODE_WORD_RE = re.compile(r"[^\W_]+")
_INJECTION_MARKERS = re.compile(
    r"""["'](?:score|refused|confidence|should_refuse)["']\s*:"""
    r"|ignore\s+(?:all\s+|any\s+)?(?:the\s+)?(?:previous|prior|above)\s+instructions"
    r"|^\s*(?:system|assistant)\s*:"
    rf"|\u2039\s*/?\s*(?:{'|'.join(DELIMITER_TAGS)})\b",
    re.IGNORECASE | re.MULTILINE,
)
_STEM_LENGTH = 5
_STOPWORDS = frozenset(
    [
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "by",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "at",
        "as",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "from",
        "which",
        "what",
        "who",
        "whom",
        "how",
        "when",
        "where",
        "than",
        "then",
        "any",
        "all",
        "each",
        "per",
        "not",
        "no",
        "nor",
        "but",
        "if",
        "so",
        "such",
        "into",
        "onto",
        "over",
        "under",
        "about",
        "after",
        "before",
    ]
)


# Function words only: a word that carries meaning (doit, peut, sauf) stays content.
# Written without accents, as words are compared after folding.
_STOPWORDS_FR = frozenset(
    [
        "les",
        "des",
        "une",
        "uns",
        "unes",
        "aux",
        "par",
        "pour",
        "sur",
        "sous",
        "dans",
        "avec",
        "sans",
        "entre",
        "vers",
        "chez",
        "est",
        "sont",
        "etre",
        "ete",
        "etait",
        "ont",
        "avait",
        "avoir",
        "sera",
        "que",
        "qui",
        "quoi",
        "dont",
        "quel",
        "quelle",
        "quels",
        "quelles",
        "lequel",
        "laquelle",
        "lesquels",
        "son",
        "sa",
        "ses",
        "leur",
        "leurs",
        "mon",
        "mes",
        "ton",
        "tes",
        "notre",
        "nos",
        "votre",
        "vos",
        "cet",
        "cette",
        "ces",
        "ceci",
        "cela",
        "celui",
        "celle",
        "ceux",
        "celles",
        "ils",
        "elle",
        "elles",
        "nous",
        "vous",
        "lui",
        "eux",
        "et",
        "ou",
        "mais",
        "donc",
        "car",
        "comme",
        "aussi",
        "ainsi",
        "alors",
        "puis",
        "selon",
        "lors",
        "tout",
        "tous",
        "toute",
        "toutes",
        "plus",
        "moins",
        "tres",
        "oui",
        "non",
        "pas",
    ]
)


def _fold(text: str) -> str:
    """Lower case without accents: "Réalisé" and "realise" compare equal."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


def content_words(text: str, *, language: Language = "en") -> set[str]:
    """Lower-cased content words of ``text``: stopwords and one- or two-letter words
    dropped, words cut to their first five letters, numbers kept whole."""
    if language == "fr":
        tokens, stopwords = _UNICODE_WORD_RE.findall(_fold(text)), _STOPWORDS_FR
    else:
        tokens, stopwords = _WORD_RE.findall(text.lower()), _STOPWORDS
    words: set[str] = set()
    for word in tokens:
        if word.isdigit():
            words.add(word)
        elif len(word) > 2 and word not in stopwords:
            words.add(word[:_STEM_LENGTH])
    return words


def has_injection_marker(text: str) -> bool:
    return _INJECTION_MARKERS.search(text) is not None


def suspected_documents(hits: list[RetrievalHit]) -> set[str]:
    """Ids of documents with at least one retrieved chunk carrying an injection marker.

    Chunk text is checked after the delimiter neutralisation the prompts apply, so a
    raw ``</document>`` in a document counts as a marker too.
    """
    return {
        str(h.chunk.document_id) for h in hits if has_injection_marker(neutralise(h.chunk.text))
    }


def citation_coverage(
    answer: str, citations: tuple[Citation, ...], *, language: Language = "en"
) -> float:
    """Share of ``answer``'s content words found in the cited quotes; 1.0 when the
    answer has none."""
    answer_words = content_words(answer, language=language)
    if not answer_words:
        return 1.0
    quoted = content_words(" ".join(c.quote for c in citations), language=language)
    return len(answer_words & quoted) / len(answer_words)


def best_cited_rerank_score(
    citations: tuple[Citation, ...], hits: list[RetrievalHit]
) -> float | None:
    cited = {c.chunk_id for c in citations}
    scores = [h.score for h in hits if h.kind is RetrieverKind.RERANKED and h.chunk.id in cited]
    return max(scores) if scores else None


@dataclass(frozen=True)
class SupportCheck:
    coverage: float
    cited_rerank_score: float | None
    reason: RefusalReason | None

    @property
    def rationale(self) -> str:
        if self.reason is RefusalReason.SUSPECTED_INJECTION:
            return "A cited document contains text that tries to instruct the model."
        if self.reason is RefusalReason.LOW_CITATION_COVERAGE:
            return (
                f"Only {self.coverage:.0%} of the answer's content words appear in its "
                "cited quotes."
            )
        if self.reason is RefusalReason.IRRELEVANT_CITATION:
            return (
                f"The cited passages score {self.cited_rerank_score:.2f} for relevance "
                "to the question, below the configured minimum."
            )
        return "The cited quotes cover the answer and are relevant to the question."


def check_support(
    answer: str,
    citations: tuple[Citation, ...],
    hits: list[RetrievalHit],
    *,
    min_coverage: float,
    min_rerank_score: float,
    language: Language = "en",
) -> SupportCheck:
    coverage = citation_coverage(answer, citations, language=language)
    rerank = best_cited_rerank_score(citations, hits)
    suspected = suspected_documents(hits)
    reason: RefusalReason | None = None
    if any(str(c.document_id) in suspected for c in citations):
        reason = RefusalReason.SUSPECTED_INJECTION
    elif coverage < min_coverage:
        reason = RefusalReason.LOW_CITATION_COVERAGE
    elif rerank is not None and rerank < min_rerank_score:
        reason = RefusalReason.IRRELEVANT_CITATION
    return SupportCheck(coverage=coverage, cited_rerank_score=rerank, reason=reason)


__all__ = [
    "SupportCheck",
    "best_cited_rerank_score",
    "check_support",
    "citation_coverage",
    "content_words",
    "has_injection_marker",
    "suspected_documents",
]
