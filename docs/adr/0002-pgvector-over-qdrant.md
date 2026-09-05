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
