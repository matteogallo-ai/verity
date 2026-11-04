"""Gold chunks of an eval question, resolved from its evidence quotes.

An answerable question lists, in ``evidence``, verbatim quotes from the corpus that
together answer it; its gold chunks are the chunks containing one of them
(whitespace-normalised). Chunk ids follow the character offsets of the parse, and
docling's layout model does not parse a PDF identically on every CPU, so ids committed
on one machine need not exist on another while the quotes still do: the eval resolves
them against the chunks it ingested. Plain substring lookup on purpose: labelling the
retriever with the retriever would defeat the benchmark.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from verity.types import Chunk, ChunkId, EvalExample


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def gold_chunk_ids(
    evidence: Sequence[str], chunks: Sequence[Chunk]
) -> tuple[list[ChunkId], list[str]]:
    """The ids of the chunks holding a quote, quote by quote in chunk order without
    repeats, and the quotes no chunk holds."""
    texts = [(c.id, normalise(c.text)) for c in chunks]
    ids: list[ChunkId] = []
    missing: list[str] = []
    for quote in evidence:
        hits = [cid for cid, text in texts if normalise(quote) in text]
        if not hits:
            missing.append(quote)
        ids.extend(h for h in hits if h not in ids)
    return ids, missing


def resolve_gold(
    examples: Sequence[EvalExample], chunks: Sequence[Chunk]
) -> tuple[list[EvalExample], list[str]]:
    """Each example with evidence gets the gold chunks of ``chunks``; an example without
    evidence keeps the ids it carries. Returns the examples and one problem per quote
    no chunk holds: a gold set silently short of a chunk would inflate recall misses."""
    out: list[EvalExample] = []
    problems: list[str] = []
    for example in examples:
        if not example.evidence:
            out.append(example)
            continue
        ids, missing = gold_chunk_ids(example.evidence, chunks)
        problems.extend(f"{example.id}: no chunk contains {quote!r}" for quote in missing)
        out.append(example.model_copy(update={"relevant_chunk_ids": tuple(ids)}))
    return out, problems
