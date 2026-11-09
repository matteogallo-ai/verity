# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project
follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- Scorecard schema v7: the run's answerable / unanswerable counts move from `dataset`
  to `sample_sizes.n_answerable_run` / `n_unanswerable_run`. `dataset` describes the
  file (`test.jsonl`: 56 questions); the run of record completed 54 of them (36 + 18).

## [1.1.0] — 2026-11-07

### Added

- Invented answers (`answer_quality.invented_answer_rate`): judged answers with a claim
  the judge finds unsupported, or no usable claim, over every judged answer, answerable or
  not. It is the README key result; the answerable-only hallucination rate stays next to it.
- Correctness against `reference_answer`, graded apart from faithfulness
  (`scripts/judge_correctness.py`, `scorecards/correctness.json`).
- κ bootstrap interval of the judge (`scripts/judge_agreement.py kappa-ci`), refusal
  recall per tag (`scripts/refusal_by_tag.py`), exact McNemar on paired runs
  (`verity.eval.paired`).
- Replay of the refusal checks one at a time on the committed runs, with no model call
  (`scripts/gate_ablation.py`, `scorecards/gate-ablation.json`).
- Pre-registered comparison of Verity against a naive RAG baseline on the 95 questions
  (`docs/preregistration/two-arm-comparison.md`, `scripts/compare_two_arm.py`,
  `scorecards/two-arm-comparison.json`): exact McNemar, Holm over three outcomes, power
  computed before the runs.
- Scorecard schema v6: `sample_sizes.n_not_run`, `cost.per_query_with_judge_usd`.

- The synthesizer reads retrieved text datamarked (spotlighting, arXiv:2403.14720):
  every run of whitespace, invisible format characters or blank look-alikes between
  words becomes `ˆ`, which `synthesize.prompt` v0.3.0 names as the mark of document
  content; a mark the model echoes is removed from the answer. Quotes are matched
  across marks and whitespace and stored as the document's own text, so a quote that
  crosses a line break is no longer dropped; such a quote keeps its line break, which
  the eval judge's `citation_ok` substring test compares as written. The confidence
  scorer is unchanged.
- `datasets/eval/injection/`: 12 poisoned documents whose instructions are written in
  plain English (none matches the injection marker list), one question each, and
  `scripts/run_injection_eval.py`, which counts the answers that ship what a poisoned
  document planted and, on the clean corpus alone, the questions still answered;
  `--no-datamark` runs the same code without the mark.
- Injection measurement, 3 live runs per arm on `claude-sonnet-4-6` ($2.16 in all),
  committed in `datasets/eval/injection/runs/` with a hand review of every marker match
  and pooled by `scripts/summarise_injection_runs.py` into
  `scorecards/injection-spotlighting.json`: 0/12 cases steered with the mark and 0/12
  without (Wilson 95 % [0, 0.243] each, the case being the unit); answers refused on
  cases the clean corpus answers, 4/24 with the mark and 7/24 without; 33/33 controls
  answered in each arm; the mark adds about 75 % input tokens (mean 4,926 against 2,822
  per answer). The set does not separate the arms.
- `verity.eval.intervals.wilson` is the one Wilson interval; the headline generator,
  the judge validation and the injection summary use it.
- Eval corpus of 21 documents of a fictional company in contracts, HR and finance;
  the PDF and DOCX files are rendered from Markdown sources in `datasets/corpus_src/`
  (`scripts/build_corpus_documents.py --check`), so every eval run goes through docling.
- 95 labelled questions (62 answerable, 33 out-of-scope) with gold chunks resolved from
  evidence quotes, and a frozen, seeded dev/test split of 39 and 56
  (`scripts/split_dataset.py --check`).
- An eval run artefact records its dataset hash, the SHA-256 of every prompt, its
  duration, its confidence threshold and, per example, the confidence score, the
  refusal reason, the judge's usage and the models that served it.
- The headline scorecard is computed from the run artefact alone (schema v6) and
  `scripts/generate_headline_results.py --check` rebuilds it; every proportion carries
  its k/n and Wilson 95 % interval, refusals are counted by recorded cause, and
  faithfulness and citation accuracy are published per question and per claim.
- Live runs `eval_6ec544d_dev.json`, on which the refusal threshold rule was fixed, and
  `eval_969e1d5_test.json` (n=56), the runs of record until the `513b839` test run (see
  Changed). A refusal
  analysis (threshold sweep, reliability diagram, recalibration) is checked against
  them, and two reruns of the dev run give a run-to-run range per metric
  (`scripts/noise_band.py`).
- Judge validation: 60 claims, 40 real and 20 perturbed to be false
  (`verity.eval.perturb`), labelled blind by a human. Judge prompt v0.1.0, the judge of
  the `6ec544d` and `969e1d5` runs, agrees with κ 0.71 (53/60). An adjudication with
  the judge's verdict visible, published apart, reaches κ 0.93 with 6 of its 7 label
  changes towards the judge; the blind result stays the result of record.
- Judge prompt v0.2.0 delimits the evidence, the answer and the citations, neutralises
  delimiter tags inside them, datamarks the evidence as `synthesize.prompt` v0.3.0
  does, and names for each claim the citation that supports it; the code sets
  `citation_ok` false when that index is missing or not one of the answer's citations.
  Rejudged on the same blind labels under a $1.30 cap, which stopped it after 52 of
  the 60 claims: κ 0.60 (44/52), against κ 0.64 (45/52) for v0.1.0 on the same claims.
  The `513b839` run of record is judged by v0.2.0.
- The API ingests the corpus and loads the embedding and reranker models at startup, in
  the background; `/ready` reports `"preload"` and stays `503` until every corpus file is
  in, and a failed preload is started again by the next `/ready`. With the stub agent and
  the memory store, the first `/ask` took 3.0–3.4 s (model load) and now takes 0.08 s,
  after `/ready` turns `200` about 3.1 s from startup (three fresh processes each, warm
  OS file cache, measured on the owner's machine with an in-process client; the
  numbers are not reproduced by any test).
- Each answer records its per-stage wall-clock (`stage_ms`) and the retries the provider
  SDKs made (`usage.retries`); both are kept per example in eval run artefacts. Retries
  are counted on calls that succeed: a call that exhausts its retries and fails over to
  the next provider records none.
- `verity ask --budget-usd` (default `VERITY_ASK_BUDGET_USD`, $1): before each LLM call,
  spend so far plus that call's estimated upper cost must fit the cap, else the answer stops
  with error kind `budget_exceeded`, exit 4, and the spend so far printed.
- `/ask` spending budget per API key over a sliding window (`VERITY_API_BUDGET_USD`, $5;
  `VERITY_API_BUDGET_WINDOW_S`, 3600 s); over budget is `429` with `Retry-After`.
- `GET /ready`, separate from `/health`: `503` unless the Postgres store answers and the
  LLM providers can serve in routing order; states only in the body, causes in the log;
  results reused for 5 s. `/health` stays liveness only.

### Changed

- The run of record is `eval_513b839_test.incomplete-20261107T122042441670Z.json`:
  shipped defaults, `synthesize.prompt` v0.3.0, judge prompt v0.2.0, 54 of 56 questions
  before its spending cap (`scorecards/live-v1.1.0.json`).
- `VERITY_IRRELEVANT_CITATION_CHECK` defaults to false: on dev it made more wrong
  refusals than right ones. `VERITY_CONFIDENCE_THRESHOLD` defaults to 0.0.
- The judge is labelled by its prompt's own validation: prompt v0.2.0 with κ 0.60 on 52
  of 60 claims, v0.1.0 with its bootstrap interval, any other prompt as unvalidated.
- An SLO counts as met only when its Wilson upper bound is.
- `SECURITY.md` names one reporting channel, GitHub private vulnerability reporting.

- No sampling parameter is sent to any model: API version 2026-10-21 rejects
  `temperature`, and the Anthropic SDK moved to 2.x. Reproducibility of the published
  numbers rests on the committed artefacts (ADR 0007), not on sampling.
- The live eval judge must be a different model from the agent's.
- The system card renders from the v1.1.0 scorecard the README uses and reads its k/n
  directly; an older scorecard schema is refused.
- Injection cases inj-02 and inj-04 have narrower success markers, since the
  21-document corpus states their former markers in clean text; the committed
  injection runs used the former markers on the 6-document corpus of `8547ae5`.
- ADR 0003 (refusal as the headline feature) is marked superseded by measurement.
- System card SLOs are approved targets, no longer proposals. A miss is published as
  **SLO not met** with its cause, and the p95 latency is reported cold start included and
  warm, in separate rows, recomputed from the run's per-answer latencies. The v1.0.0 run of
  record misses both: 32.5 s cold start included, 22.4 s warm.
- Claims C09, C24 and C25 state what they cover. Findings of a neutral
  audit of `main` and what became of each: `docs/audit/2026-11-03-neutral-audit.md`.

### Fixed

- Questions a spending cap left unrun were counted as request errors, and the cost per
  query divided by every question of the dataset; they are now `n_not_run`, outside the
  error count and the cost denominator.
- A defect inside a retrieval stage was reported as "store down" (503); it now
  propagates, the agent types it `internal_error` and the API answers 500, still
  charging what the request paid for.

- Database connection errors are logged without the Postgres password.
- The Docker image bakes in the docling models, and the smoke test uploads a PDF; one
  docling converter is shared per process.
- Tests never read the developer's `.env`.
- The confidence scorer read a refusal flag given as a string with `bool()`, so
  `"refused": "false"` counted as a refusal. A flag that is not a boolean now makes the
  response unparseable.
- A failure of the local embedding or reranking model at question time (weights not in
  the cache, out of memory) escaped the agent as a raw exception: `/ask` answered `500`,
  the decompose call it had paid for was not counted in the key's budget, and
  `verity eval run` stopped. It is now a typed error, `model_unavailable`, and `/ask`
  answers `503`.
- Re-ingesting an edited document left its old chunks in the store when the edit moved
  chunk boundaries: the stale text stayed retrievable and citable, and a quote was
  validated against text the document no longer had (claim C01). Ingestion now replaces
  a document's chunks in one transaction, in both stores; a corpus file that was deleted
  leaves the index with its chunks; a document whose text (SHA-256), chunks and
  embedding model match the index is not embedded again. `verity ingest` reports
  unchanged and removed documents. Removal matches documents by their repo-relative
  identity, so the host CLI and the API container prune each other's copies; it never
  runs on an empty corpus directory, and keeps a document whose file cannot be read or
  failed to parse. One database holds one corpus version: the last sync wins, so an
  API image built from an older checkout removes files a newer checkout added. A model
  whose weights change under the same name, or a change to document metadata only, is
  not detected; delete the rows in SQL to force a re-embed.
- An answer that fails after some LLM calls returned now reports their tokens and cost
  (it reported zero), in `Answer.usage` and in the `answer_failed` log line. Eval
  `cost.*` figures and `verity eval run --budget-usd` therefore include the spend of
  errored answers.
- Concurrent answers from one agent (the API) each report their own usage; the
  decompose usage of one could be reported by another.
- The faithfulness judge read the string `"false"` as supported; a verdict is now a JSON
  boolean or the string `true`/`false`, anything else makes the reply malformed.
- A provider with no API key is left out of the default route: an OpenAI-only setup
  answered `503` to every `/ask`. With no key at all, the router fails at construction.
- PDF and DOCX conversion runs in a worker thread; it blocked every other request.
- A judge outage during `verity eval run` errors that example instead of ending the run
  and discarding its results; the example's spend still counts against `--budget-usd`,
  and a rejected judge key stops the run.
- A paid eval run that stops at `--budget-usd` (exit 4) or loses examples to an outage
  (exit 6) writes an incomplete run, `eval_<sha>_<dataset>.incomplete-<time>.json`, with
  every paid example, errored ones included; it previously persisted nothing on a budget
  abort, kept only the ids of errored examples, and wrote the headline for a run with
  errors. Incomplete runs never overwrite a file and are left out of `eval compare`.
  Errored examples count against the cap and in the printed total.
- Delimiter tags written with a space before the slash (`< /document>`) are neutralised.
- `VERITY_DATABASE_URL` is held as a secret and no longer appears in printed settings.

### Security

- Container images are pinned by digest and tracked by Dependabot (Docker and Compose
  ecosystems); no workflow checkout persists its token.

- `/status` requires `X-API-Key`: it lists every corpus file name and the models in use.
  `/health` and `/ready` remain the public probes. The CI smoke script sends the key.
- Failed key attempts are limited per client address, per /64 for IPv6
  (`VERITY_API_AUTH_FAILURES_PER_MINUTE`, 10), then `429` with `Retry-After`, checked before the key is compared. At most 10,000
  addresses are tracked. Behind a reverse proxy, uvicorn must trust its forwarded headers
  (SECURITY.md).
- Dependencies with known advisories upgraded: Python (pillow 13.0.0, urllib3 2.8.0,
  docling 2.145.0, filelock, certifi, opentelemetry, setuptools) and web (next
  15.5.28, vitest 4.1.12, vite 6.4.5, eslint 9.39.5, postcss 8.5.32 and, for
  `@typescript-eslint/typescript-estree`, minimatch 11.0.5 via override). v1.0.1 ships
  the vulnerable versions.
- CI: actions pinned by commit SHA, read-only workflow tokens, Dependabot, CodeQL,
  an OSV-Scanner job with one dated exception (`osv-scanner.toml`: accelerate, no fix
  released, its vulnerable loaders called by nothing, checked by
  `tests/unit/test_osv_exceptions.py`) and an OpenSSF
  Scorecard workflow. See `docs/security/README.md`.

## [1.0.1] — 2026-09-08

### Fixed — headline scorecard denominators (schema v2 → v3)

**Reporting-only correction, no re-spend, no new run.** The v1.0.0
scorecard published on 2026-09-07 (tag `v1.0.0`, commit `94bd0ba`)
averaged its judge-track metrics over all 16 examples with refusals
contributing 1.0 by judge convention. That convention inflates
`citation_accuracy` : refusals get a "free" 1.0 even though the judge
never ran on them (`4 refusals bypass the judge` → 4 gratuitous 1.0
contributions pushed the aggregate up).

Concretely on the same audit trail
(`datasets/eval/runs/live/eval_d0ca1d5_questions.json`, unchanged) :

| Metric | v1.0.0 (v2) | **v1.0.1 (v3)** | Delta |
|---|---:|---:|---:|
| citation_accuracy | 0.792 (over 16) | **0.722 (answered, n=12)** | −0.070 |
| faithfulness | 1.000 | 1.000 (answered, n=12) | 0 |
| hallucination_rate | 0.000 | 0.000 (answerable_and_answered, n=11) | 0 |
| judge_model | `"(12/12)"` | **`"(12/16 answered — 4 refusals bypass the judge)"`** | wording |

The **audit trail is untouched** — every per-example `judge_claims`
record is bit-identical to what v1.0.0 shipped. The recompute from
`per_audit` gives the exact v3 numbers, proving this is a reporting
refinement, not a rerun.

### Schema v3 details

- Every judge-track metric is now a `{value, scope, n_denominator}`
  triple — a chiffre cannot be quoted without knowing what was averaged.
  Scoping rule:
  - `faithfulness` / `citation_accuracy` : scope = `answered`,
    denominator = `n_judged` (the answers the judge actually saw).
    Refusals bypass the judge and no longer inflate these aggregates.
  - `hallucination_rate` : scope = `answerable_and_answered`,
    denominator = the at-risk set (the only questions where the
    per-example formula can return 1.0).
- New top-level `sample_sizes` block (`n_examples`, `n_answered`,
  `n_judged`, `n_refused`, `n_answerable_answered`) so every
  denominator is one click away from the scorecard.
- Judge provenance summary format is now
  `"claude-sonnet-4-6 (12/16 answered — 4 refusals bypass the judge)"`
  when refusals cause skips. The v2 `"(12/12)"` was arithmetically
  true but misleading — 4 refusals had already been filtered out of
  the denominator, hidden from the reader.
- CLI console rows now display each metric with its explicit
  scope + denominator (e.g. `Citation accuracy (answered, n=12)`).
  Console-vs-JSON label consistency asserted by
  `tests/unit/test_label_source_of_truth.py`.
- New `build_headline_dict_from_eval_run(EvalRun, dataset_path, …)`
  helper lets a persisted audit trail be re-serialised into the
  current headline schema without re-running the LLM — used to
  migrate the v1.0.0 live scorecard from v2 to v3 with zero re-spend.

### Docs
- **README restructured for the 15 s CTO scan** : title + tagline →
  problem in two sentences → two real UI screenshots (cited answer +
  calibrated refusal, captured via Playwright against the running app,
  no fabrication) → scorecard table with explicit scope on every
  metric → architecture Mermaid → Quickstart → Constats Ouverts →
  Engineering discipline → Stack → Demo → UI → Dev.
- New `docs/assets/screenshot-answer.png` and
  `docs/assets/screenshot-refusal.png` — un-edited Playwright captures
  of the demo (stub agent, `make demo`), reproducible locally in ~30 s.

### Note on the v1.0.0 Release
The tag `v1.0.0` is left in place at `94bd0ba` (never retag a public
release). The v1.0.0 Release notes on GitHub carry a banner pointing
at v1.0.1 as the honest headline ; the audit trail at v1.0.0 stands.

## [1.0.0] — 2026-09-07

### S7 finale — evaluated on real Anthropic + audit-first persistence

**One live run of record**. `claude-sonnet-4-6` as agent and judge, over the
16-question labelled dataset (11 answerable + 5 out-of-scope). Total cost
$0.2713. Every chiffre in the README derives verbatim from
`scorecards/live-v1.0.0.json` + `datasets/eval/runs/live/eval_d0ca1d5_questions.json`
— no hand-typed numbers, no averaged runs, no re-shot chiffre. Committed
as immutable artefacts.

Headline (schema `verity.scorecard.headline/v3`, each judge-track metric
carries `{value, scope, n_denominator}`) :
- **refusal_precision 1.000, refusal_recall 0.800** — the differentiator
- faithfulness 1.000 (answered, n=12), citation_accuracy 0.722 (answered, n=12),
  hallucination_rate 0.000 (answerable_and_answered, n=11)
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

### Added — headline scorecard v3 schema (pre-tag correction)
- Every judge-track metric is now a `{value, scope, n_denominator}` triple —
  no chiffre can be quoted without knowing what was averaged. Scoping rule:
  - `faithfulness` / `citation_accuracy` : scope = `answered`,
    denominator = `n_judged` (the answers the judge actually saw). Refusals
    bypass the judge and no longer inflate these aggregates with a
    convention 1.0. **Material effect on this run: `citation_accuracy`
    corrects from 0.792 (over 16, refusals=1.0) to 0.722 (over 12
    answered)** ; faithfulness stays 1.000 (all 12 judged were 1.0 too).
  - `hallucination_rate` : scope = `answerable_and_answered`,
    denominator = the at-risk set (the only questions where the
    per-example formula can return 1.0). 0.000 on this run either way.
- New top-level `sample_sizes` table (`n_examples`, `n_answered`,
  `n_judged`, `n_refused`, `n_answerable_answered`) so every denominator
  is one click away from the scorecard.
- Judge provenance summary format is now
  `"claude-sonnet-4-6 (12/16 answered — 4 refusals bypass the judge)"`
  when refusals cause skips. The v2 `"(12/12)"` was arithmetically true
  but misleading — 4 refusals had already been filtered out of the
  denominator. Load-bearing honesty rule: a reader cannot confuse the
  judge call count with the total question count.
- CLI console rows now display each metric with its explicit
  scope + denominator (e.g. `Citation accuracy (answered, n=12)`).
  Console-vs-JSON label consistency asserted by
  `tests/unit/test_label_source_of_truth.py`.
- New `build_headline_dict_from_eval_run(EvalRun, dataset_path, ...)`
  helper lets a persisted audit trail be re-serialised into the current
  headline schema without re-running the LLM — used to migrate the v1.0.0
  live scorecard from v2 to v3 with zero re-spend.

### Added — headline scorecard v2 schema (superseded by v3 above)
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
