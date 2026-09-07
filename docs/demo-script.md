# 90-second demo script — v1.0.0

Beat-for-beat script matching the v1.0.0 thesis :
**upload → cited answer → calibrated refusal → measured**.

No voiceover. Silent-captioned. Recorded from a real session or omitted — no fake
GIF, no re-shoot to nudge a chiffre.

## Frame

```bash
make demo          # FastAPI (stub) on :8000 + Next dev on :3000, both offline
```

## Beats

### 0:00–0:10 — the claim
Open on the README's headline table.
Highlight `refusal_recall 0.800`, `refusal_precision 1.000`,
`hallucination_rate (answerable only) 0.000`.
Overlay caption : *"RAG that refuses when the evidence is weak — and proves it. Every number below comes from one live run, committed as an audit artefact."*

### 0:10–0:35 — upload → cited answer
Drop `sample-msa.pdf` on the UI.
Ingestion progress bar → chunk count → embedding done.
Type : *"What is the notice period for terminating the master services agreement?"*
Answer appears with inline citation. Click the citation → source pane
highlights the exact clause. Confidence panel shows `1.000` + which chunks
were used + rerank scores. Provenance badge reads `claude-sonnet-4-6`.

### 0:35–0:55 — calibrated refusal (the differentiator)
Same corpus, ask an out-of-scope question :
*"What is the company's parental leave policy?"*
System refuses with a concrete rationale : *"The Employee Handbook and all
other evidence documents contain no mention of parental leave policy…"*
No fabricated answer, no null citation, no fluent hallucination. This is
the money shot.

### 0:55–1:25 — the eval track
Terminal :

```bash
uv run verity eval run --ci                  # deterministic stub, seconds
uv run verity eval compare                   # regression diff across commits
cat scorecards/live-v1.0.0.json | jq         # committed audit of the live run of record
```

Show :
- The scorecard's per-example `per_refusal` (4 TP refusals, 1 FN — q-015 documented in Constats).
- The `per_audit` array — for every answer, the raw text, citations, judge's per-claim verdict.
- The `provenance` block — `agent=claude-sonnet-4-6 (16/16)`, `judge=claude-sonnet-4-6 (12/12)`, commit sha, dataset sha256.

### 1:25–1:30 — close
Architecture diagram (Mermaid in the README) → repo link.
Overlay caption : *"Every chiffre in the README derives from `scorecards/live-v1.0.0.json`
+ `datasets/eval/runs/live/…` — no hand-typed numbers, no averaged runs."*
