# ADR 0002 — pgvector over a dedicated vector database

**Status:** accepted · 2026-09

## Context
The spec allows pgvector or Qdrant. Verity needs, in one system: dense vector search,
sparse/keyword search (BM25/FTS), chunk + document metadata, and persisted eval-run
history for regression tracking.

## Decision
Use **Postgres + pgvector** as the single stateful service.

## Rationale
- **One-command local run is a hard requirement.** `docker compose up` with a single
  stateful container is materially simpler to reason about and to demo than app + vector
  DB + relational DB. Fewer moving parts is itself a maturity signal.
- Postgres already provides the *sparse* half of hybrid retrieval (FTS / `ts_rank`,
  or BM25 via extension), so dense and sparse live behind one connection — no second
  system to keep consistent.
- Metadata, gold eval labels, and scorecard history are relational data that belong in a
  relational store. Colocating them with the vectors keeps a query's whole footprint in
  one transaction boundary.
- At the corpus sizes Verity targets (enterprise document sets, not web-scale), pgvector's
  performance is not the bottleneck; retrieval *quality* and *measurement* are.

## Consequences
If a future workload outgrows pgvector's ANN performance, the `VectorStore` protocol
(`verity.retrieval.base`) is the single seam to reimplement against Qdrant — the rest of
the pipeline is unaffected. This is deliberately isolated behind one interface.

## Amendment — 2026-11: eval history is versioned JSON, not Postgres rows

The rationale above names "scorecard history" as relational data to colocate with the
vectors. That was never built: eval runs are JSON files written by
`verity.eval.run_store.FileRunStore`, and this amendment records that as the decision.

- **Runs of record** (`datasets/eval/runs/live/`) and the published scorecard
  (`scorecards/`) are committed. Git is their history: each one is tied to the commit
  that produced it, reviewed in the PR that adds it, and readable without a database.
- **Local and CI runs** (`datasets/eval/runs/eval_*.json`, `retrieval_*.json`) are
  ignored by git; `verity eval compare` diffs any two of them.
- A stub run cannot overwrite a live run at the same path (`StubOverwritesLiveError`).

Rejected: a `runs` table. It would make the audit trail depend on a database a reviewer
does not have, and a fresh clone would hold no history. The cost of the choice: no SQL
across runs; at a few dozen runs, `jq` over the files is enough.

Postgres still holds documents, chunks, vectors and the full-text index, and since the
same month it is the default store for the API, the demo and the eval (claim C22).
