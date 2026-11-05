# The five false refusals of the Verity arm on dev (naive-baseline comparison)

Read from the pinned artefacts only; no model was called. Verity arm:
`eval_e644a98_dev.incomplete-20261105T094431636600Z.json`; naive arm:
`eval_e644a98_dev-naive-baseline.incomplete-20261105T094823388125Z.json`. Both ran the
same `synthesize` prompt (sha256 `3a68ba00…` in both artefacts), so the naive arm's
draft is what Verity's synthesizer writes for the same question, up to sampling.

## Which check refused

On the 21 answerable questions both arms completed, Verity refused 5. All 5 were
refused by the deterministic support check (`verity.agent.support`), before the
scorer: each made 2 LLM calls (decompose, synthesize), not 3, so neither the scorer
nor the threshold took part. None was a `no_citation` refusal.

| Id | Question (short) | Check | Recorded value | Gold chunk retrieved | Naive draft, same prompt |
|---|---|---|---|---|---|
| q-001 | MSA notice period | `irrelevant_citation` | best cited rerank −0.62 | yes, and it is the cited chunk | "60 days … Amendment No. 1"; 3/3 claims supported |
| q-023 | Northwind data hosting | `irrelevant_citation` | best cited rerank −2.45 | yes, and it is the cited chunk | "exclusively in Frankfurt"; 2/2 supported |
| q-022 | Northwind payment term vs MSA | `low_citation_coverage` | coverage 47 % | yes (ranks 1, 2) | "45 days … longer than the MSA's 30"; 3/3 supported |
| q-031 | End-user training in the SOW | `low_citation_coverage` | coverage 31 % | yes (rank 0) | "No … explicitly excluded"; 3/3 supported |
| q-035 | London hotel limit | `low_citation_coverage` | coverage 44 % | yes (rank 0) | "US$325 per night"; 2/2 supported, 1 citation not accepted |

The cited chunk of an `irrelevant_citation` refusal is identified by its score: the
rationale's value equals the reranker score of exactly one hit, and that hit is a
chunk listed in the question's `relevant_chunk_ids`.

## Causes

**Relevance (q-001, q-023): the right chunk, judged as a whole.** Both cited chunks
pack two sections, and the answer is in the second:

- the q-001 chunk opens with "1. Limitation of Liability", then "2. Termination for
  Convenience … sixty (60) days";
- the q-023 chunk opens with "3. Billing and Payment", then "4. Data Residency … hosted
  exclusively in … Frankfurt".

The cross-encoder scores the chunk against the question, and the first section pulls
the score below 0.

**Coverage (q-022, q-031, q-035): the measure, not the evidence.** The answers restate
the question ("nightly hotel limit", "end-user training") or state a relation the
quotes do not word ("longer than", "explicitly excluded"). Those content words are
missing from the quotes. The naive drafts of the same questions recompute, with
`citation_coverage`, to 45 %, 45 % and 57 %: correct answers sit on both sides of 0.5.

## Too strict, or a real lack of evidence?

Not a lack of evidence: in all five, the gold chunk was retrieved, and in the two
relevance refusals it is the very chunk cited. The judge finds every claim of the
same-prompt naive drafts supported.

Whether lowering a threshold would help depends on what the gate also stops. Below
are all refusals by these two checks in the three Verity live runs, on dev
(`6ec544d`, `e644a98`) and test (`969e1d5`):

| Check | Answerable refused (false) | Unanswerable refused (correct) |
|---|---|---|
| coverage < 0.5 | 7, at 31–47 % | 10, at 17–48 % |
| cited rerank < 0.0 | 9, at −2.45 to −0.01 | 2, at −8.72 and −0.11 |

- **Coverage: no threshold separates the two columns.** Lowering it lets through
  unanswerable drafts at the same values as the answerable ones it recovers.
- **Relevance: the setting is too strict.** The 0.0 was chosen on the 16 examples of
  `d0ca1d5` and never validated (comment on `min_cited_rerank_score` in
  `src/verity/config.py`). On dev alone, every answerable it refused scored between
  −2.45 and −0.62. It stopped one unanswerable at −8.72; the other, at −0.11, is above
  every answerable value.

What this cannot show:

- **What would follow if a draft passed the gate.** The scorer would then run. That
  takes a paid run to know.
- **Verity's own drafts.** A refused answer is stored as "I don't know" with no
  citations, so its draft and quotes are not in the artefact; the naive drafts are a
  proxy.
- **Generalisation.** The counts are small (7, 10, 9, 2).

Nothing here changes a setting. The completion runs of the amendment of 2026-11-05
(`docs/preregistration/naive-rag-baseline.md`) keep the registered configuration. A
change to the relevance threshold or the chunking would be its own pre-registered run
on dev.

## Reproduce

```python
import json
from verity.agent.support import citation_coverage
from verity.types import Citation

naive = (
    "datasets/eval/runs/live/eval_e644a98_dev-naive-baseline.incomplete-20261105T094823388125Z.json"
)
rows = {r["example_id"]: r for r in json.load(open(naive))["per_audit"]}
for q in ("q-001", "q-022", "q-023", "q-031", "q-035"):
    r = rows[q]
    print(
        q,
        round(citation_coverage(r["answer_text"], tuple(Citation(**c) for c in r["citations"])), 2),
    )
```

The per-check counts come from `refusal_reason` and the number in `refusal_rationale`
of each refused row of the three Verity artefacts.
