# Evaluation dataset

`questions.jsonl` holds the labeled examples used by `verity eval run`: 19 questions
over the 6 documents in `datasets/corpus/` (11 answerable, 8 out-of-scope, 3 of them
adversarial). One JSON
object per line:

```json
{
  "id": "q-001",
  "question": "…",
  "answerable": true,
  "reference_answer": "…",
  "relevant_chunk_ids": ["ef55252e-…"],
  "tags": ["single-hop"]
}
```

- `reference_answer` is `null` when `answerable` is false.
- `relevant_chunk_ids` are the gold chunks for retrieval metrics. Chunk ids are
  deterministic (uuid5 of a path-independent document identity plus the chunk's
  ordinal and offsets, see `verity.ingestion.ids`), so they are stable
  across machines. They are filled for every answerable example and empty for
  out-of-scope ones. `scripts/build_eval_labels.py` regenerates them;
  `tests/integration/test_eval_labels_integrity.py` checks they resolve.
- `tags`: `single-hop`, `multi-hop` (needs decomposition), `numeric`, `near-miss`
  (close to the corpus but not answerable from it), `out-of-scope`, `adversarial`
  (out of scope, and `vendor_onboarding_note.md` holds instructions telling the
  model to answer it with a score of 0.99; the right outcome is a refusal).

The committed live run (`runs/live/eval_d0ca1d5_questions.json`) predates the
adversarial document and questions: it covers q-001 to q-016 over 5 documents.

Out-of-scope examples measure whether Verity refuses. Definitions of every metric
are in `docs/evaluation-methodology.md`.

`runs/` holds run artefacts: `runs/live/` the live runs of record (committed),
`runs/eval_*.json` stub runs (ignored by git), `runs/retrieval_*.json` retrieval
scorecards.
