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

**Relevance (q-001, q-023): the right chunk, scored as a whole.** A hit keeps the best
cross-encoder score of its text over the decomposer's sub-questions (a single-fact
question is its own only sub-question). The local model reproduces both recorded
scores from (question, chunk text). Both chunks
hold more than the answer:

- q-001: the chunk is "## 1. Limitation of Liability" followed by "## 2. Termination for
  Convenience … sixty (60) days". Without §1 it scores +3.29 instead of −0.62.
- q-023: the chunk starts mid-§3 ("Fees are invoiced annually…"; "3. Billing and
  Payment" is only its section metadata) and holds §4 "Data Residency … hosted
  exclusively in … Frankfurt", §5 and the heading of §6. Without the §3 text it still
  scores −0.45; §4 alone scores +0.51.

So the score of an answer-bearing chunk drops below 0 when the chunk also carries
sections the question is not about.

**Coverage (q-022, q-031, q-035): the measure, more than the evidence.** Verity's
refused drafts are not stored, so the evidence is the naive drafts of the same
questions. Those restate the question ("nightly hotel limit", "end-user training") or
state a relation the quotes do not word ("longer than", "explicitly excluded").
`citation_coverage` puts them at 45 %, 45 % and 57 %: correct answers sit on both sides
of 0.5. Two limits on that proxy:

- For q-022 and q-031 the two arms decomposed differently and retrieved different hits.
- In the q-035 naive draft, two of the three uncovered words ("excluding taxes") belong
  to a claim the judge marked as not supported by its citation, so coverage did catch
  one uncited claim there.

## Too strict, or a real lack of evidence?

Not a lack of evidence: in all five, the gold chunk was retrieved, and in the two
relevance refusals it is the very chunk cited. The judge finds every claim of the
same-prompt naive drafts supported.

Whether lowering a threshold would help depends on what the gate also stops. Below
are all refusals by these two checks in the three Verity live runs, on dev
(`6ec544d`, `e644a98`) and test (`969e1d5`). Counted in refusals; in brackets, distinct
questions, since q-022, q-023 and q-031 were refused in both dev runs:

| Check | Answerable refused (false) | Unanswerable refused (correct) |
|---|---|---|
| coverage < 0.5 | 7 (5), at 31–47 % | 10 (10), at 17–48 % |
| cited rerank < 0.0 | 9 (8), at −2.45 to −0.01 | 2 (2), at −8.72 and −0.11 |

The relevance row only holds drafts that passed coverage, which is checked first; the
relevance score of a coverage-refused draft is not recorded.

- **Coverage: no threshold separates the two columns.** Lowering it lets through
  unanswerable drafts at the same values as the answerable ones it recovers.
- **Relevance: the setting is too strict.** The 0.0 was chosen on the 16 examples of
  `d0ca1d5` and never validated (comment on `min_cited_rerank_score` in
  `src/verity/config.py`). On dev alone, every answerable it refused scored between
  −2.45 and −0.62. It stopped one unanswerable at −8.72; the other, at −0.11, is above
  every answerable value on dev.

What this cannot show:

- **What would follow if a draft passed the gate.** The scorer would then run. That
  takes a paid run to know.
- **Verity's own drafts.** A refused answer is stored as "I don't know" with no
  citations, so its draft and quotes are not in the artefact; the naive drafts are a
  proxy.
- **Generalisation.** The counts are small.
- **A clean test split for a new threshold.** The table above reads refusal values from
  the test run `969e1d5`. A relevance threshold chosen after reading them cannot be
  validated on that run; it needs a fresh held-out measurement.

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
