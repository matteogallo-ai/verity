# Evaluation dataset

- `questions.jsonl`: 95 labeled questions over the 21 documents in `datasets/corpus/`
  (see `datasets/CORPUS.md`): 59 answerable, 27 unanswerable, 9 adversarial.
- `dev.jsonl` / `test.jsonl`: the frozen stratified split (39 / 56, seed in
  `scripts/split_dataset.py`). Tune on dev, publish on test.

One JSON object per line:

```json
{
  "id": "q-001",
  "question": "...",
  "answerable": true,
  "reference_answer": "...",
  "evidence": ["verbatim quote from the corpus", "..."],
  "tags": ["multi-hop"],
  "relevant_chunk_ids": ["ef55252e-..."]
}
```

- `reference_answer` and `evidence` are `null` / empty when `answerable` is false.
- `relevant_chunk_ids` are generated: `scripts/build_eval_labels.py` marks every chunk
  that contains one of the `evidence` quotes. Chunk ids are deterministic (uuid5 of a
  path-independent document identity plus the chunk's ordinal and offsets, see
  `verity.ingestion.ids`), so they are stable across machines.
  `tests/integration/test_eval_labels_integrity.py` checks labels and split are current.
- `tags`: `single-hop`, `multi-hop` (evidence in at least two documents), `negation`,
  `numeric`, `out-of-corpus`, `near-miss` (close to the corpus but not answerable from
  it), `false-premise`, `adversarial` (prompt injection in the question, or q-017 to
  q-019, whose retrieved document `vendor_onboarding_note.md` holds the injection).

Editing a question: change `questions.jsonl`, then run `build_eval_labels.py` and
`split_dataset.py`. Adding questions re-draws the split; do it before tuning, never after.

The committed live run (`runs/live/eval_d0ca1d5_questions.json`) predates this dataset:
it covers 16 questions over 5 documents.

`runs/` holds run artefacts: `runs/live/` the live runs of record (committed),
`runs/eval_*.json` stub runs (ignored by git), `runs/retrieval_*.json` retrieval
scorecards. Metric definitions: `docs/evaluation-methodology.md`.
