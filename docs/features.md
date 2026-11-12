# Features and roadmap

What Verity does, each with the claim that backs it, and what it does not do yet.

## Features

Each feature maps to its implementation and tests in [`CLAIMS.md`](CLAIMS.md); CI job
`claims` runs those tests, and `scripts/check_claims.py` fails when a bullet below has no
claim, or a claim's status disagrees with the section it is listed in.

- Verbatim citations: every quote is a substring of its source document; others are dropped (C01).
- Refusal with a rationale when a citation check fails, the scorer refuses or its score is below a threshold (C02); unparseable scorer output is refused (C03).
- Hybrid retrieval: dense + sparse, reciprocal rank fusion, cross-encoder rerank (C04).
- Two vector stores: in-memory with cosine and BM25 (C05); Postgres + pgvector with full-text search (C06), the default for the API, the demo and the eval (C22).
- Markdown and plain-text ingestion with structure-aware chunking and machine-independent ids (C07); PDF and DOCX through docling (C08).
- Multi-provider LLM routing with fallback, stopping on authentication errors (C09).
- Per-request provenance: each answer records the model that actually served it (C10).
- Evaluation harness: retrieval, refusal and LLM-judged answer-quality metrics, one audit record per question (C11).
- Live run artefacts protected from being overwritten by stub runs (C12).
- 95 questions over 21 documents (≥ 25 unanswerable, ≥ 5 adversarial) with a frozen, seeded dev/test split; per-question and per-claim aggregates (C28).
- Results table (`docs/results.md`) and the README key result generated from the scorecard and checked in CI (C13); CLI labels match scorecard keys (C14).
- SDK call signatures (C15) and LLM output parsers (C16) tested against the shapes real models produce.
- Per-stage tracing behind `/metrics`, `/metrics/prom` and `/dashboard` (C17); optional OTLP export, bounded in-memory spans (C29).
- HTTP API with `/status`, `/ask` and content-addressed `/ingest` (C18), and a Next.js UI with citation highlighting and a provenance badge (C19).
- Provider API keys held as secrets, absent from any printed settings (C20).
- `docker compose up`: non-root image, Postgres, migrations, corpus ingestion and the API in order, smoke-tested in CI (C26).
- `make demo` stays on the stub agent, at no cost, even with a provider key in `.env` (C38).
- `make demo-cct CORPUS=<folder>`: one collective agreement served in French with the settings measured on such texts, each passage under an article heading labelled with it, with the one-click questions listed in the folder's `suggested_questions.json`; the model is called only with `LIVE=1` (C39).
- `VERITY_SUPPORT_CHECK=figures`: an answer is refused when it states a figure that neither a passage it cites nor the question contains; the words it takes from the question no longer count against it (C40).
- API hardening: API keys on `/status`, `/ask` and `/ingest`, per-key rate limit, a limit on failed key attempts, bounded inputs, file-type allow-list, generic errors with a request ID (C23).
- Bounded spend: per-key spending budget on `/ask`, a per-answer cap on `verity ask` checked before each LLM call, bounded `max_tokens` and timeouts (C31).
- Liveness and readiness split: `/health` answers while the process lives; `/ready` answers `503` until the corpus and the local models are loaded at startup, and unless Postgres answers and the LLM providers can serve as routing uses them, without spending tokens (C32).
- Retrieved text reaches the synthesizer datamarked (spotlighting): a mark between every pair of words lets its prompt label all of it as document content. Against 12 naturally worded poisoned documents, 3 live runs per arm on the earlier 6-document corpus, no case was steered with or without the mark (0/12 each, Wilson 95 % [0, 0.243]), though the attacks made answerable questions refuse (4/24 answers with the mark, 7/24 without): this set does not show an effect of the mark (C33).
- Prompt-injection defence: delimited, untrusted-labelled evidence, and a deterministic support check (citation coverage, cited relevance, injection markers) the model's score cannot override; adversarial eval questions (C24).
- Provider and store outages reported as errors (HTTP 503), never as refusals (C25).
- Naive RAG baseline: the same pipeline with every check disabled (`--naive-baseline`), compared with Verity by an exact McNemar test on the same questions, from pinned artefacts (C35).
- Verity against the naive baseline on all 95 questions, pre-registered with a power analysis: exact McNemar per question, Holm-adjusted over three primary outcomes ([results](results.md#verity-against-a-naive-rag-baseline-95-questions), C37).
- Correctness against the reference answers, refusal recall per question tag, and the judge's κ with a bootstrap interval, each regenerated from its artefact (C36).
- Judge validated against blind human labels on 60 claims, real and perturbed to be false: κ, agreement, specificity and sensitivity with n and intervals, generated from a pinned artefact (C27).

## Roadmap

Not implemented, or implemented without a test. Each item is a `roadmap` row in
[`CLAIMS.md`](CLAIMS.md).

- Web-page ingestion: the trafilatura parser exists but no test exercises it (C30).
- Calibrated refusal: the measured test ECE (0.116) does not support it (C21).
- A judge from a different model family, and a second human annotator (C34).
