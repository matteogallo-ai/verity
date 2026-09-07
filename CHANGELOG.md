# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project
follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.0.0] — 2026-09-07

### S7 finale — evaluated on real Anthropic + audit-first persistence

**One live run of record**. `claude-sonnet-4-6` as agent and judge, over the
16-question labelled dataset (11 answerable + 5 out-of-scope). Total cost
$0.2713. Every chiffre in the README derives verbatim from
`scorecards/live-v1.0.0.json` + `datasets/eval/runs/live/eval_d0ca1d5_questions.json`
— no hand-typed numbers, no averaged runs, no re-shot chiffre. Committed
as immutable artefacts.

Headline (schema `verity.scorecard.headline/v2`) :
- **refusal_precision 1.000, refusal_recall 0.800** — the differentiator
- faithfulness 1.000, citation_accuracy 0.792, hallucination_rate (answerable only) 0.000
- retrieval nDCG@8 0.953, recall@8 1.000, precision@k 0.148 (k>gold ceiling)

### Added — audit trail persistence
- `PerExampleAudit` model (question, answer_text, refused, refusal_rationale,
  citations, hits_used, `judge_claims` with per-claim `supported`/`citation_ok`,
  per-example `AnswerMetrics`, `UsageStats`) persisted verbatim under
  `EvalRun.per_audit`. Every published aggregate is now traceable back to the
  exact answer + judge verdict that produced it, without re-spending on the
  judge. Contract locked by `tests/unit/test_audit_persistence.py`.
- Enriched headline scorecard JSON at `scorecards/live-v1.0.0.json` — full
  provenance triple + per-request agent/judge summaries + `dataset.sha256` +
  `total_cost_usd` broken into agent + judge.

### Added — audit-write-protected paths
- **Path separation**: live-provenance runs (any run where agent OR judge is a
  real LLM) auto-nest into `runs_dir/live/`. Stub runs stay at plain
  `runs_dir/`. Neither CI, nor `verity eval run --ci`, nor
  `generate_headline_results --check` ever touches `live/`. Truth-table
  proof for all 4 (judge, agent) combos in
  `tests/unit/test_stub_live_path_separation.py`.
- **`StubOverwritesLiveError` / `StubOverwritesLiveScorecardError`** in the
  run store and headline writer as a belt-and-braces safeguard. Bypassable
  only via explicit `allow_overwrite=True`. Prompted by an incident where
  a routine `--ci` invocation destroyed a live audit trail; the class of
  accident is closed.

### Added — CLI + observability
- `--report-only-floors` flag on `verity eval run` : for the one-shot LIVE
  run, floor breaches log as constats without blocking (exit 0) so a run of
  record is never marked as "failed" in a way that invites a re-spend.
- `--budget-usd MAX` hard cap on cumulative cost. Aborts (exit 4) mid-run
  if the cap is exceeded — safety net for the one-shot run.
- `--scorecard-json PATH` writes the enriched headline JSON alongside the
  regular `EvalRun` file. **Written unconditionally BEFORE floor evaluation**
  so a breach never destroys the artefact.
- Console verdict `FN (hallucinated on out-of-scope)` renamed to
  `FN (didn't refuse out-of-scope)` — the label was doing work the code
  didn't back up (confusion-matrix cell, not judge verdict).

### Added — headline scorecard v2 schema
- `answer_quality.hallucination_rate` → `answer_quality.hallucination_rate_answerable`
  (same number, honest name — the metric is gated on `example.answerable`).
- New companion `refusal_calibration.out_of_scope_answered` : `{count, total}`.
  Covers the half the answerable-gated metric cannot see.
- New CLI console row for the companion, matching JSON labels 1:1. Console
  and JSON vocabulary locked by
  `tests/unit/test_label_source_of_truth.py` — no more drift possible
  between the two surfaces.

### Fixed — anthropic SDK 1.x + prompt drift + scorer parser
- **`anthropic` 1.x removed `temperature`** from `AsyncMessages.create` typed
  kwargs. `AnthropicClient.complete()` now passes it via `extra_body`. Pinned
  `anthropic>=1.0,<2.0` and `openai>=3.0,<4.0`.
  SDK-signature smoke tests in `tests/unit/test_llm_sdk_schema.py` bind the
  kwargs to the real signatures per LLM call site — a future SDK bump that
  renames or drops a kwarg fails CI, not the live spend.
- **Confidence scorer refused 16/16 on first live attempt** because the
  prompt didn't render the JSON schema (it was in a `#` comment stripped by
  the PromptLang parser). Added the schema to the user block, mirroring the
  synthesize prompt. Same fix applied to the judge prompt. Parsers gained
  tolerance for common alias keys (`confidence`/`should_refuse`/`reasoning`,
  `atomic_claims`/`is_supported`/`citation_supports`) as defence in depth.
  23 real-Sonnet-shape parser tests in `tests/unit/test_llm_output_parsers.py`.
- **Refusal-path provenance invariant**: `RagAgent.__init__` gains
  `preferred_provider` / `preferred_model` populated from the routing
  client; `Answer.provider_used` falls back to that when a synthesis
  bypass returns unstamped. `api.state.provenance_for` returns
  `agent_model="unknown"` instead of the boot fallback when a synthesizer
  bypass yields no stamp — no silent claim that a run served a live model
  when it actually did not.

### Added — README, demo, engineering discipline
- README rewritten with **results table first**, generated from the
  scorecard JSON via `scripts/generate_headline_results.py` (never
  hand-typed, idempotent, fails loudly on missing markers). Architecture
  Mermaid diagram (GitHub-native), 90-second demo script at
  `docs/demo-script.md`, Constats Ouverts section.
- README `Engineering discipline` section surfaces the path-separation
  guarantee, the safeguard, and the "one run of record" invariant — real
  guarantees, not narrative.

## [0.7.0] — 2026-09-07

### S6 micro-patch (pre-tag audit) — honesty tightening

- **Provenance is now per-request, not boot-time.** `Answer` gains optional
  `provider_used: Provider | None` + `model_used: str | None`, stamped by
  `RagAgent` from the synthesizer's `Completion`. `AppRuntime.provenance_for
  (answer)` derives `Provenance` from that stamp. When S7 live routing lands,
  a `ProviderUnavailableError` fallback (Anthropic → OpenAI) is reflected
  honestly per request instead of showing the boot-time primary model.
  `AskResponse.answer` now also carries `provider_used` + `model_used` on the
  wire. New test: `test_ask_provenance_reflects_the_actual_provider_used`.
- **Confidence readout is labelled + shows signals.** New UI component
  `ConfidenceReadout` renders `answer.confidence.score` verbatim (passthrough,
  no client-side composite) with an explicit provenance label
  (`agent-reported (stub: stub-agent)` or `agent-reported (<model>)`) and the
  underlying signals a careful reader would want: `refused`, `citations` count,
  top rerank score, evidence hits count. Never a bare authoritative percentage
  next to an answer. Both `AnswerCard` and `RefusalState` embed the readout.
  +5 vitest cases covering the labelling and signal exposure.
- **Verbatim citation invariant enforced by test.** New
  `tests/unit/test_citation_verbatim_invariant.py` walks the whole answerable
  question set and asserts
  `document.text[citation.char_start:citation.char_end] == citation.quote`
  on every emitted citation. Stacks on top of the existing synthesizer
  substring guard, so any future refactor that lets a paraphrased citation
  through breaks CI.

### Added
- **S6 — Next.js UI (`web/`) + thin HTTP wrappers.**
  - **Backend endpoints (additive serializer layer)** — three thin wrappers
    over the same runtime the CLI uses (:mod:`verity.runtime`):
    - `GET /status` — real provenance triple (`agent_model`, `is_stub`,
      `embedding_model`, `reranker_model`), real corpus counts. No hard-coded
      `"ok"` — honest state of the process.
    - `POST /ask` — returns `AnswerView` (verbatim mirror of
      :class:`Answer` / :class:`Citation` / :class:`UsageStats` /
      :class:`Confidence`) with `provenance` adjacent so a chiffre can't be
      lifted without its badge. Every `hits_used[i]` carries a
      structurally-derived `cited` flag (chunk id ∈ citations set).
    - `POST /ingest` — multipart upload. Document id is **content-addressed**:
      `uri = upload://<sha256><ext>`, `doc_id = uuid5(NAMESPACE_URL, uri)`.
      Same bytes under a different filename → same id (idempotent re-uploads).
  - **`verity.runtime`** — shared composition helpers so the CLI and the API
    build identical component graphs. Zero business logic reimplemented in
    the HTTP layer.
  - **`verity.api.state.AppRuntime`** — process-scoped singleton owning the
    in-memory store + embedder + reranker + agent, with corpus preload on
    first request. Configured via a `packageManager`-style boot toggle
    (`agent_mode="stub"|"live"`; auto-picks `stub` when no key is set).
  - **Web UI** (`web/`):
    - Next.js 15 App Router, TypeScript strict, ESLint 9 flat config, pnpm@9.15.0
      pinned via `packageManager`, Node 22.
    - `AnswerCard` renders the answer with inline `[N]` citation markers that
      map to source chunks; refusal is a first-class visual state
      (`RefusalState`), never an error.
    - `SourcePanel` shows every retrieved chunk with its rerank score, kind,
      section, and cited/not-cited badge — all values verbatim from the API.
    - `ProvenanceBadge` is rendered next to every answer AND in the topbar;
      `StubModeBanner` is persistent iff `provenance.is_stub` on the API
      response (same condition the tests assert against).
    - Typed API client (`lib/api.ts`); `NEXT_PUBLIC_API_BASE` (default
      `http://localhost:8000`). No secrets client-side.
  - **Demo path**: `make demo` boots the FastAPI stub-mode server + Next.js
    dev server together on `:8000` + `:3000`. Ctrl-C stops both cleanly.
  - **CI**: new `web` job (Node 22 · corepack pnpm@9.15.0 · frozen lockfile)
    runs lint + typecheck + vitest + `next build`. The 3 existing Python
    jobs (`quality`, `integration`, `evaluation`) are byte-preserved.

### Honesty guarantees (rules of iron — enforced by tests)
- No value on screen is fabricated. Every latency, score, cited flag, and
  confidence originates in the API response for that specific request.
- **Provenance is adjacent to every answer.** `AskResponse` never returns an
  answer without its `provenance` triple. The badge sourced from
  `agent_model` renders next to the answer text.
- Refusal is a state, not an error — the UI has a dedicated
  `RefusalState` component with rationale + hits shown for auditability.
- Content-addressed upload IDs prevent the S1.1 non-portable-id class of bug
  from re-emerging in the upload path.
- Stub mode is loudly visible: yellow banner across the top whenever the API
  reports `provenance.is_stub === true`; yellow badge on every answer card.

### Notes
- No new eval numbers published. Retrieval + refusal chiffres shown in the
  UI come from the exact same `answer.confidence` and `answer.hits_used`
  fields the eval harness already surfaces — S7 will add the live-judge
  headline numbers.
- No Playwright/E2E in S6 (flake budget). Candidate for S7.

## [0.6.0] — 2026-09-07

### Added
- **S5 — observability + latency de-biasing.**
  - **Tracer stack.** `NoOpTracer` is the default (no dependency, no overhead
    — keeps every existing test green) ; `OTelTracer` wraps the OpenTelemetry
    SDK and exports every span through an `InMemorySpanExporter` the FastAPI
    layer queries. When `settings.otel_endpoint` is set, spans are additionally
    forwarded to an OTLP HTTP collector.
  - **Span coverage.** `RagAgent.answer` opens one span per stage
    (`agent.decompose`, `agent.retrieve`, `agent.synthesize`,
    `agent.confidence`). `HybridRetriever` refines the retrieve stage into
    `retrieve.embed_query`, `retrieve.dense`, `retrieve.sparse`,
    `retrieve.fuse`, `retrieve.rerank`. Each span carries `latency_ms`,
    `usage.*` (tokens + cost when relevant), and retrieval hit counts +
    top-score.
  - **Structured logging with trace_id.** `configure_logging()` wires
    `structlog.contextvars.merge_contextvars` so every log line inside a
    request carries the current `trace_id`. `request_trace(trace_id)` is the
    single binding point (used by `RagAgent.answer`).
  - **Warmup + `Scorecard.cold_start_ms`.** `AgentEvaluator(warmup=True)`
    (default) runs one throwaway `agent.answer("warmup …")` to prime lazy
    components before the timed loop starts. The measured wall-clock of that
    priming call lands in `Scorecard.cold_start_ms` as a distinct field ; the
    reported `latency.p50/p95/p99` reflects steady-state serving. `--no-warmup`
    is available for the cold-start-inclusive measurement.
  - **`GET /metrics` (JSON) + `GET /metrics/prom` (Prometheus text) + `GET
    /dashboard` (server-rendered HTML)**. FastAPI app; `verity serve` boots
    it via uvicorn. The dashboard shows live session metrics from the tracer
    + persisted eval-run history — no raw chunk text or prompt ever exposed.

### Measured latency de-biasing (git SHA d05355b, `--judge stub`, `agent=stub-agent`)
Before (v0.5.0, no warmup):
- p50 = 89.4 ms · p95 = **1534.7 ms** · p99 = **4992.6 ms** · `cold_start_ms` unavailable
- p95/p99 dominated by first-invocation embedder+reranker model loads.

After (v0.6.0, warmup ON):
- p50 = 89.7 ms · **p95 = 104.8 ms** (×15) · **p99 = 107.4 ms** (×46)
- `cold_start_ms = 5264.2` reported separately.

Retrieval, refusal, and answer-quality chiffres are byte-identical
(`retrieval nDCG=0.950`, `refusal_recall=1.000`, `hallucination_rate=0.0`) —
the warmup only changes the operational metrics, never the eval verdicts.

### Notes
- No new dependencies added (opentelemetry, fastapi, uvicorn were already
  pinned in S0).
- The 116 prior unit tests are byte-preserved; +14 new S5 unit tests
  (`test_tracer`, `test_trace_context`, `test_warmup`, `test_api_server`).
  New integration test `test_api_serve.py` exercises the full FastAPI stack.

## [0.5.0] — 2026-09-06

### Added
- **S4 — eval harness + first full Scorecard.**
  - **Corpus expansion.** Three new documents (`datasets/corpus/dpa.md`,
    `sla.md`, `employee_handbook.md`) — DPA (Schedule B referenced by MSA §3.2),
    SLA (uptime + credits + response times + maintenance), and internal
    employee handbook (PTO, working model, resignation). Dataset expanded from
    6 → **16 questions** (11 answerable, 5 out-of-scope of which **3 near-miss**),
    every question carries a `reference_answer`, all IDs auto-generated by
    `scripts/build_eval_labels.py` (deterministic, repo-relative).
  - **`FaithfulnessJudge`** protocol implementations:
    - `LLMFaithfulnessJudge` — real judge via `RoutingClient` +
      `judge_faithfulness.prompt`. Requires an API key at runtime.
    - `StubFaithfulnessJudge` — deterministic mechanical judge for CI and for
      local runs without a key. Every scorecard produced by a stub run carries
      `judge_model="stub-judge-v1"` so the faithfulness numbers are
      distinguishable from a real LLM verdict.
  - **Refusal calibration** (`verity.eval.refusal`) — deterministic confusion
    matrix from (expected_answerable × observed_refused) pairs, yields
    `refusal_precision` + `refusal_recall`. Zero LLM calls: this chiffre is real
    from day one, in every mode.
  - **`AgentEvaluator`** (`verity.eval.harness`) — runs the agent on the whole
    dataset and aggregates a full `Scorecard`: retrieval (macro-averaged),
    answer-quality (from injected judge), refusal (deterministic), latency
    p50/p95/p99, cost/query mean. `HarnessResult` also carries the full
    per-example breakdown for later inspection.
  - **`FileRunStore`** (`verity.eval.run_store`) — JSON per (SHA, dataset)
    under `datasets/eval/runs/eval_<sha>_<dataset>.json`, with
    `history()` / `latest()` and a `diff_runs` helper that classifies each
    metric delta as regression or improvement based on the metric's "better"
    direction.
  - **CLI**:
    - `verity eval run --judge stub|live` — produces the Scorecard, distinguishes
      REAL from STUB chiffres explicitly in the terminal output, persists an
      `EvalRun`, supports `--floor-refusal-recall`, `--floor-ndcg`, `--floor-recall`
      for CI gates. `--judge live` triggers the S3.1 missing-key guard.
    - `verity eval compare` — diffs the two most recent runs of a dataset, prints
      a Rich table with ↓ regressions in red / ↑ improvements in green, and exits
      with code 3 when any metric regressed.
  - **CI**: eval job replaces the S0 scaffold command with a real
    `verity eval run --judge stub --floor-refusal-recall 0.8 --floor-ndcg 0.5
    --floor-recall 0.75`.
- **Scripted stub agent extended** with 7 new themes (SLA uptime + credit, DPA
  location + SCC, retention, P1 response, PTO, office days, maintenance) so the
  eval harness produces meaningful chiffres on all 11 answerable questions.
  Score-based theme matcher disambiguates keyword collisions (e.g. "notice"
  now clearly belongs to maintenance, not termination).

### Measured retrieval (git SHA 72d47b9, `--judge stub`, `agent=stub-agent`, in-memory backend, 16 questions)
- Retrieval precision@8 = 0.148 (capped by k>gold size), recall@8 = 1.000,
  nDCG@8 = 0.950. **Real chiffres** — deterministic, embeddings + reranker are
  the real components used in production.
- Latency p50 / p95 / p99 = 84 / 1477 / 4784 ms (in-memory backend, stub LLM;
  p95/p99 dominated by first-invocation model loads).

### Refusal calibration — measured on stubbed agent
- `refusal_precision = 1.000`, `refusal_recall = 1.000` on the shipped run:
  5/5 out-of-scope refused (incl. 3 near-miss: stock options, cloud provider
  name, parental leave), 11/11 answerable answered.
- **The maths is deterministic (no LLM call).** The observed refuses come from
  the *scripted stub agent* the CI harness uses (`agent_model="stub-agent"` on
  the persisted run) — so these chiffres reflect the stub's calibration, not
  Verity's real behaviour under a live LLM. A `--judge live` run against a
  real agent produces the real-agent refusal calibration alongside faithfulness.
- The Scorecard, the persisted JSON, and the CLI output all carry
  `agent_model` so a refusal chiffre can never be lifted without the caveat.

### Faithfulness / citation accuracy / hallucination rate — pending real judge
Persisted for traceability but explicitly labelled `judge_model=stub-judge-v1`
in the scorecard artefact and in the CLI output. Not published as headline
numbers in the README until a run with a real LLM judge is executed out-of-band.

### Provenance triple
Every `EvalRun` now carries three model labels: `embedding_model` (retrieval),
`agent_model` (LLM in the agent), `judge_model` (LLM-as-judge). The CLI header,
the persisted JSON, and the README table all render them explicitly so no
chiffre can be interpreted without its provenance.

### `verity eval compare` — compatibility filter + documented exit codes
- Two runs are comparable **iff** they share the same `dataset`, `agent_model`,
  and `judge_model`. A stub-agent run is never diff'd against a live-agent run
  (any refusal/latency delta would reflect the provenance change, not a real
  regression); same for stub-judge vs real-judge. Incompatible runs are skipped
  with an explicit "nothing comparable" message.
- Exit codes are documented in the command's help and stable:
  - **0** — no regression. Includes the "fewer than 2 compatible runs" case
    ("nothing to compare"), so a fresh checkout with a single persisted
    scorecard exits cleanly.
  - **3** — at least one metric regressed beyond epsilon (worse than the
    metric's "better" direction).
- Unit tests are hermetic: they seed a `tmp_path` runs directory via
  `--runs-dir` rather than reading the real `datasets/eval/runs/`, so the
  suite is deterministic regardless of what the local checkout has committed.

### Notes
- The README's `_pending S4_` placeholders are replaced by real retrieval
  numbers; refusal numbers are shown alongside their `agent=stub-agent`
  provenance; faithfulness / citation / hallucination stay
  `_pending real judge_`.

## [0.4.0] — 2026-09-06

### Added
- **S3 — agentic RAG with citations + calibrated refusal.**
  - **LLM runtime** (`verity.llm`): `AnthropicClient`, `OpenAIClient` (vendor
    errors translated to `AuthenticationError` / `ProviderUnavailableError`),
    `StubLLMClient` (scripted, offline, deterministic). Thin `.prompt` adapter
    (per ADR 0004) parses the shipped PromptLang sources and renders them into
    `Message` lists with strict variable substitution.
  - **`MultiProviderRoutingClient`**: honours `settings.provider_order`.
    `ProviderUnavailableError` triggers ordered fallback; **`AuthenticationError`
    short-circuits** (no silent fallback across a bad key — audit-trail intent,
    covered by a dedicated test).
  - **Agent stack** (`verity.agent`):
    - `LLMQuestionDecomposer` — parses JSON list, degrades to `[question]` on
      malformed output, cap at `settings.max_agent_steps`.
    - `CitedSynthesizer` — draft answer + validated `Citation`s. Every citation's
      `quote` is verified to be a verbatim substring of the referenced chunk and
      the offsets are re-anchored to `document.text` (`chunk.char_start + pos`).
      Hallucinated citations are dropped.
    - `LLMConfidenceScorer` — calibrated confidence + refusal against
      `settings.confidence_threshold`; malformed scorer output forces refusal
      (fail-open on principle).
    - `RagAgent` orchestrator: decompose → per-sub-q retrieve → aggregate hits →
      synthesize → score → refuse-or-ship. Refusal is a first-class output
      (empty citations, honest "I don't know" text, rationale, hits_used
      preserved for traceability).
  - **`prompts/synthesize.prompt`** — new PromptLang source for cited-answer
    synthesis (JSON schema `{answer, citations:[{quote, chunk_index}]}`).
  - **CLI `verity ask`** — question → cited answer + confidence + rationale +
    usage table. `--stub` flag runs the scripted responder end-to-end without
    keys (same script the integration test asserts against).
- **README**: one line under "Measured results" pointing to
  `datasets/eval/runs/` so reviewers can find the actual retrieval scorecards
  without waiting for S4.

### Notes
- No faithfulness / refusal numbers are published in this release — that
  scorecard is S4. The retrieval scorecard from 0.3.0 remains the only
  measured chiffre in the repo.

## [0.3.0] — 2026-09-06

### Added
- **S2 — hybrid retrieval + first measured scorecard.**
  - `ReciprocalRankFusion` (Fusion protocol): score = Σ 1/(rrf_k + rank_i),
    dedup by `chunk.id`, deterministic tie-break on chunk id for
    bit-reproducible fusion output.
  - `CrossEncoderReranker` (Reranker protocol): local
    `cross-encoder/ms-marco-MiniLM-L-6-v2`, batched, async via
    `asyncio.to_thread`, backend factory injectable (unit lane never loads torch).
  - `HybridRetriever` (Retriever protocol): embed query → dense ∥ sparse
    (`asyncio.gather`) → RRF → cross-encoder rerank. Structured `stage`
    logs per phase (`dense`/`sparse`/`fused`/`reranked` + top score + count).
    Robustness: a stage error or empty sparse degrades to `[]` without crashing.
  - `InMemoryVectorStore` — second implementation of the `VectorStore` protocol,
    backed by numpy cosine + `rank_bm25`. Makes the unit lane exercise the full
    hybrid pipeline offline and lets `verity eval retrieval --in-memory`
    produce a scorecard on any laptop with no Postgres.
  - Retrieval metrics: `precision_at_k`, `recall_at_k`, `ndcg_at_k` with binary
    gain + macro-averaging (`BinaryRetrievalMetric` implements the
    `RetrievalMetric` protocol from `eval.base`). Undefined-on-empty-gold cases
    raise explicitly rather than silently returning NaN.
  - CLI:
    - `verity retrieve "…"` — one-shot retrieval, Rich table of top hits.
    - `verity eval retrieval` — real measured retrieval scorecard tagged to
      the current git SHA, persisted as JSON under `datasets/eval/runs/`.
      Supports `--floor-precision / --floor-recall / --floor-ndcg` for CI
      anti-regression gates.
  - CI integration lane adds `verity eval retrieval --no-in-memory
    --floor-ndcg 0.5 --floor-recall 0.75` as a hard gate.

### Measured retrieval scorecard (in-memory backend, git SHA 82ebaa5, k=8, 4 answerable questions)
- precision@8: **0.156** (capped by k > gold size; 1-2 gold chunks / 8 slots)
- recall@8: **1.000**
- nDCG@8: **0.877**
- per-example nDCG: q-001 0.631 · q-002 0.877 · q-003 1.000 · q-005 1.000

The pgvector run will produce equivalent numbers (same embedder, same
reranker, same fusion) — will be re-verified in the first CI run on 0.3.0.

## [0.2.0] — 2026-09-05

### Fixed
- **Portable IDs (S1.1).** `document_id_for_uri` now normalises its input through a
  new `identity_for` helper: local file paths are hashed as their POSIX path
  *relative to the repo root* (e.g. `datasets/corpus/msa.md`), not as absolute
  machine-specific paths. `Document.uri` still stores the absolute path for I/O and
  provenance. Consequence: `datasets/eval/questions.jsonl` chunk ids are now stable
  across machines and CI runs (regenerated via `scripts/build_eval_labels.py`).

### Added
- `tests/integration/test_eval_labels_integrity.py`: asserts every
  `relevant_chunk_id` resolves to a chunk containing the expected anchor phrase,
  and that unanswerable questions keep an empty label set. Would have caught the
  non-portable id bug on the first CI run.
- Chunker emits a structured `chunk_over_budget` WARNING when an emitted chunk
  exceeds `max_words` (defensive guard against silent embedding truncation).

### Added
- **S1 — ingestion & indexing.**
  - Structure-aware `Parser` implementations: native TXT + Markdown (heading hierarchy
    preserved), docling for PDF/DOCX (headings, sections, pages captured), httpx +
    trafilatura for web pages. `SourceTypeParser` dispatches by URI/extension.
  - `StructureAwareChunker` packs paragraphs into ~512-token chunks along section
    boundaries with light overlap. Load-bearing invariant:
    `document.text[chunk.char_start:chunk.char_end] == chunk.text` (asserted on the
    example corpus).
  - Deterministic identifiers: `Document.id = uuid5(NAMESPACE_URL, uri)` and
    `Chunk.id = uuid5(NAMESPACE_URL, f"{document.id}:{ordinal}:{cs}:{ce}")`. Re-ingesting
    the same source produces the same ids — the vector store upsert is idempotent and
    eval labels stay stable.
  - `LocalEmbedder` around sentence-transformers (default `BAAI/bge-small-en-v1.5`,
    384-dim). Batched, wrapped in `asyncio.to_thread`, dim-checked against
    `config.embedding_dim`, backend factory is injectable so unit tests avoid model
    downloads. **The model weights download on the first run** — plan ~130 MB plus
    torch (~2 GB via `uv sync`).
  - `PgVectorStore` (pgvector): idempotent `upsert(chunks)` and `upsert_documents`,
    cosine `dense_search`, `websearch_to_tsquery` + `ts_rank` `sparse_search`,
    `get_chunks(ids)`. HNSW index on `embedding vector_cosine_ops`, GIN on the
    generated `tsvector`. Migrations live in `migrations/`, applied by a small runner
    (advisory-locked, per-file transactions) via `verity db upgrade`.
  - `verity ingest <path|dir|url>` wired end-to-end (parse → chunk → embed → upsert),
    with a Rich summary and continue-on-error per source (structured logs via
    structlog).
- **Example corpus + eval labels.** `datasets/corpus/msa.md` (30-day termination
  clause, $1M liability cap tied to Acme Corporation as data-processing entity,
  no-competitor assignment clause) and `datasets/corpus/financial_summary_fy2024.md`
  (revenue + YoY figure). `scripts/build_eval_labels.py` reruns the ingestion pipeline
  in dry-run mode and writes `relevant_chunk_ids` for q-001/002/003/005; q-004/006
  remain out-of-scope by design (see ADR 0003).
- **CI integration lane.** `.github/workflows/ci.yml` adds an `integration` job that
  starts a `pgvector/pgvector:pg16` service, runs `verity db upgrade`, and executes
  `pytest -m integration` — so the round-trip test runs on every push without a local
  Postgres. Unit lane (`-m "not integration"`) stays DB-free and free.

### Notes
- No fabricated eval numbers — the harness lands in S4. The `pending S4` placeholders
  in the README remain in place.
- Retriever/Fusion/Reranker are still S2. This release only implements the
  `VectorStore` primitives.

## [0.1.0] — S0 scaffold

### Added
- **S0 — scaffold.** Typed domain model (`verity.types`), stage protocols
  (ingestion, retrieval, llm, agent, eval, observability), typed settings,
  CLI surface (`verity`, `verity eval run/compare`), Docker Compose (app + pgvector),
  CI pipeline (lint · typecheck · test · eval gate), docs skeleton with ADRs,
  evaluation methodology, and a dataset specification including unanswerable questions.

_No implementation logic yet — S0 defines the contracts every later stage fills._

## Roadmap
- S1 ingestion & indexing · S2 hybrid retrieval + RRF + rerank · S3 agentic pipeline
  (decomposition, citations, confidence, refusal) · S4 eval harness + headline scorecard
  · S5 observability · S6 Next.js UI (v1.1) · S7 README + demo + `v1.0.0` tag.
