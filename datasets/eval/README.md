# Evaluation dataset

Labeled examples used by `verity eval run`. Schema per line of `questions.jsonl`:

```json
{
  "id": "q-001",
  "question": "…",
  "answerable": true,
  "reference_answer": "…" ,          // null when answerable = false
  "relevant_chunk_ids": [],           // gold chunk ids, filled once the corpus is ingested (S1)
  "tags": ["single-hop"]
}
```

The set intentionally mixes four kinds (see `docs/evaluation-methodology.md`):
`single-hop`, `multi-hop` (needs decomposition), `numeric` (tables/figures), and
`out-of-scope` (unanswerable — the answer is **not** in the corpus).

`out-of-scope` examples are the point: they measure whether Verity refuses correctly.
`relevant_chunk_ids` are populated in S1 once the bundled example corpus is ingested and
chunk ids are stable; until then they are empty and retrieval metrics run against the
answerable subset only.
