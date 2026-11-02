# Threat model — OWASP Top 10 for LLM Applications 2025

Each item: whether it applies to Verity and why, the mitigation **as the code behaves**
(module and function), the tests that prove it, and what is **not** stopped. A
mitigation without a test says "no test". Test ids are `path::name`; `(slow)` tests load
the local models, `(integration)` tests need Postgres; every other test runs in the fast
lane. Claim ids (C01…) refer to [`CLAIMS.md`](CLAIMS.md); a tested behaviour with no
claim id is not advertised in the README. Deployment controls (keys, limits, bind
address) are summarised in [`SECURITY.md`](../SECURITY.md); this page is the per-threat
analysis and SECURITY.md links here.

## Scope and trust boundaries

Verity is single-tenant: one API process, one corpus, a list of API keys that are all
equal ([ADR 0006](adr/0006-api-security-baseline.md)). The parties:

| Party | Trusted for | Not trusted for |
|---|---|---|
| Operator (environment, Postgres, `datasets/corpus/`) | everything | — |
| API key holder | asking questions, adding documents | the content of what they upload |
| Document text (shipped or uploaded) | nothing: it is data | any instruction it contains |
| Question text | nothing: it is data | any instruction it contains |
| LLM provider (Anthropic, OpenAI) | producing text | the format and truth of that text |
| Unauthenticated network client | `/health`, `/status` (whose first call also triggers the corpus preload), `/metrics`, `/metrics/prom`, `/dashboard`, `/docs`, `/openapi.json` | `/ask`, `/ingest` |

## Summary

| Item | Applies | Status |
|---|---|---|
| [LLM08 Vector and embedding weaknesses](#llm08-vector-and-embedding-weaknesses) | yes | partial: write access gated, no read isolation |
| [LLM01 Prompt injection](#llm01-prompt-injection) | yes | partial: serving path tested; eval judge prompt undefended |
| [LLM02 Sensitive information disclosure](#llm02-sensitive-information-disclosure) | yes | partial |
| [LLM03 Supply chain](#llm03-supply-chain) | yes | partial, no test |
| [LLM04 Data and model poisoning](#llm04-data-and-model-poisoning) | data: yes; model: no | partial |
| [LLM05 Improper output handling](#llm05-improper-output-handling) | yes | mitigated on the server; UI rendering untested |
| [LLM06 Excessive agency](#llm06-excessive-agency) | no tools | not applicable beyond a fixed pipeline |
| [LLM07 System prompt leakage](#llm07-system-prompt-leakage) | low | prompts are public by design |
| [LLM09 Misinformation](#llm09-misinformation) | yes | partial |
| [LLM10 Unbounded consumption](#llm10-unbounded-consumption) | yes | partial: no spending cap on the API |

## LLM08 Vector and embedding weaknesses

**Applies.** Retrieval decides which text the model reads and which text the API
returns, so whoever controls the index controls both.

**The stores.** `verity.retrieval.pgvector_store.PgVectorStore` (default): tables
`documents` and `chunks` (`migrations/001_initial.sql`), each chunk holding its
plaintext, a `VECTOR(384)` embedding (HNSW, cosine) and a generated `tsvector`.
`verity.retrieval.memory_store.InMemoryVectorStore` (`VERITY_STORE=memory`): the same
data in process memory, lost at restart. Embeddings are computed locally
(`verity.ingestion.embedder.LocalEmbedder`); they are never sent to a provider and no
endpoint returns them.

**Who can write to the index.**

| Path | Gate | Test |
|---|---|---|
| `POST /ingest` | `X-API-Key` (`api.guards.require_api_key`), 30 requests a minute per key, shared with `/ask`, 1 MiB, type allow-list on extension + MIME + content (`api.guards.check_upload`), 200-document cap (`api.state.AppRuntime.ingest_bytes`) | `tests/unit/test_api_hardening.py::test_missing_key_is_401`, `tests/unit/test_api_hardening.py::test_two_megabyte_upload_is_413`, `tests/unit/test_api_hardening.py::test_disallowed_upload_type_is_415`, `tests/unit/test_api_hardening.py::test_corpus_document_limit_is_409` (C23); `tests/unit/test_corpus_cap.py::test_concurrent_uploads_cannot_overshoot_the_cap` |
| First request of a process | ingests every file under the corpus directory (`AppRuntime.preload_corpus`) | `tests/unit/test_api_router.py::test_status_reports_real_provenance_and_corpus` (C18) |
| `verity ingest`, `docker compose` ingest step | shell or container access | `tests/unit/test_container.py::test_compose_runs_db_then_migrate_then_ingest_then_app` (C26) |
| Direct SQL | Postgres credentials; Compose publishes Postgres on the host's `127.0.0.1` only | `tests/unit/test_container.py::test_published_ports_bind_loopback_only` (C26) |

An upload's id is `uuid5` of `upload://<sha256><ext>`, so an upload cannot replace a
document with different bytes or a shipped corpus file
(`tests/unit/test_api_router.py::test_ingest_upload_uses_content_addressed_id`, C18). The
store refuses an embedding dimension the schema cannot hold
(`tests/integration/test_pgvector_store.py::test_store_refuses_an_embedding_dim_the_schema_cannot_hold`
(integration), C06).

**Poisoned documents that get retrieved.** Retrieval itself has no defence: a
document is ranked on similarity alone. The defence is downstream, in
`verity.agent.support.check_support`: a draft citing any document in which a
retrieved chunk carries an injection marker is refused (`suspected_injection`), as is a
draft its quotes do not cover or whose citations the reranker scores irrelevant
(`tests/unit/test_support.py::test_a_marker_anywhere_in_the_document_distrusts_every_cited_chunk`,
`tests/unit/test_adversarial.py::test_an_obeyed_injection_does_not_ship_an_invented_answer`,
`tests/unit/test_adversarial.py::test_an_echoed_injection_is_refused_with_real_models`
(slow), C24).

**Not stopped.**

- **No isolation between readers.** The schema has no tenant or owner column and no
  query filters on one; every key holder retrieves from every document. `/ask` returns
  the full text of every retrieved chunk (`hits_used`, `api.models.answer_to_view`),
  refused answers included, so a key holder can read the corpus chunk by chunk. No
  control, no test: single-tenancy is the stated deployment model.
- **No per-document access control.** No document carries readers; no test.
- **Any key holder can poison every other client's answers.** An upload persists in
  Postgres across restarts, and no endpoint deletes it. A document that states a false
  fact relevant to a question, with no instruction-like text, passes every check and is
  answered from. A successful upload writes no log line (`verity.runtime.ingest_uri`
  logs nothing, unlike the batch pipeline), so an uploaded document cannot be traced
  back to the key that added it. No control, no test.
- **Crowding.** Up to the cap, near-duplicate uploads can fill the top-k (20 per
  retriever, 8 after reranking) and push the right passage out. No control, no test.
- **Embedding inversion** is moot rather than mitigated: the plaintext sits next to the
  vector in the same row, so anyone who can read the embeddings can read the text.
  Postgres access is the boundary.
- **Same-dimension model swap.** Only the dimension is checked. Re-pointing
  `VERITY_EMBEDDING_MODEL` at another 384-dimension model mixes incompatible vectors
  silently; `chunks.embedding_model` is stored but no query filters on it. No test.

## LLM01 Prompt injection

**Applies.** Indirect injection through ingested documents is the main risk (any key
holder can add one); direct injection through the question is the second.

**Mitigation.** `verity.agent.evidence.render_evidence` puts each chunk inside
`<document>` elements within one `<evidence>` element, and `neutralise` turns every
delimiter tag in untrusted text into an inert character, so a document cannot close
its block. The question goes through `neutralise` in the decomposer, synthesizer and
scorer, and the scorer sees the draft answer inside `<draft_answer>`. The system
section of `prompts/decompose.prompt`, `synthesize.prompt` and `confidence.prompt`
says the delimited text is untrusted data. Then `verity.agent.support.check_support`
refuses, before the scorer runs, on injection markers, low citation coverage or
irrelevant citations; the scorer's score cannot override it
(`verity.agent.agent.RagAgent._answer`).

**Tests.** `tests/unit/test_prompt_injection.py::test_a_document_cannot_close_its_delimiters`,
`tests/unit/test_prompt_injection.py::test_every_prompt_marks_its_inputs_untrusted`, `tests/unit/test_prompt_injection.py::test_a_section_title_cannot_inject_attributes_or_tags`,
`tests/unit/test_prompt_injection.py::test_the_scorer_prompt_delimits_the_draft_answer`;
`tests/unit/test_support.py::test_injection_markers_are_detected`;
`tests/unit/test_adversarial.py::test_an_obeyed_injection_does_not_ship_an_invented_answer`,
`tests/unit/test_adversarial.py::test_an_echoed_injection_is_refused_with_real_models` (slow) (C24).

**Not stopped.**

- An injection paraphrased past the marker list in `verity.agent.support`.
- The thresholds and the marker list were chosen on the same eval questions they are
  checked on.
- **The eval judge prompt is undefended.** `prompts/judge_faithfulness.prompt` places
  the retrieved context, the answer and the citations under plain `CONTEXT:` /
  `ANSWER:` / `CITATIONS:` headings, with no delimiters and no untrusted-content
  instruction, and `test_every_prompt_marks_its_inputs_untrusted` covers only the three
  serving prompts. A corpus document can therefore address the judge and move the
  published answer-quality metrics. Eval only, never on the serving path. No
  mitigation, no test.

## LLM02 Sensitive information disclosure

**Applies.** The corpus may hold confidential text, the provider receives it, and the
process holds credentials.

**Mitigation.** Provider and API keys are `SecretStr`, absent from the printed settings
(`tests/unit/test_config.py::test_api_keys_never_appear_in_settings_repr`, C20). Error
bodies are generic and the detail goes to the log under a request id
(`tests/unit/test_api_hardening.py::test_corrupt_pdf_is_a_generic_422_with_a_request_id`,
`tests/unit/test_api_hardening.py::test_unhandled_error_is_a_generic_500_with_a_request_id`, C23); FastAPI's default 422
body, which echoes the input, is replaced (`api.server._install_request_context`; no
test). Logs
carry `principal=key-<n>`, never the key
(`tests/unit/test_request_log.py::test_the_api_log_line_carries_the_request_id`). The
agent's log lines carry decision, stages, tokens and cost, not the question (no test).

**Not stopped.**

- Every key holder reads every document (see LLM08); `/status` lists corpus file names
  without a key.
- The question and the top retrieved chunks are sent to the configured provider. No
  redaction, no test.
- The Postgres DSN, password included, is a plain `str` (`Settings.database_url`) and
  appears in `repr(Settings())`. No test.
- `answer_done` logs a refusal's `rationale`, which is model-written and can quote a
  document; `answer_failed` logs the provider's exception text. No test bounds what
  they contain.
- No test asserts that an API key never appears in a log line; only the principal is
  checked.

## LLM03 Supply chain

**Applies.** Python and npm dependencies, two local models from Hugging Face, docling's
layout and OCR models, and two hosted LLM APIs.

**Mitigation.** Dependencies are pinned in `uv.lock` and `web/pnpm-lock.yaml`. The
Docker image embeds the embedding and reranker models and runs with
`HF_HUB_OFFLINE=1` (no test) as an unprivileged user
(`tests/unit/test_container.py::test_runtime_stage_runs_as_an_unprivileged_user`, C26).
The fast test lane cannot open a network socket
(`tests/unit/test_network_guard.py::test_fast_lane_test_cannot_open_an_inet_socket`).
Provider SDK call signatures are tested against the pinned SDKs
(`tests/unit/test_llm_sdk_schema.py`, C15).

**Not stopped.** No automated vulnerability scan of the lockfiles. Outside the image
the models are fetched by name with no pinned revision
(`ingestion.embedder._default_backend_factory`, `retrieval.reranker._default_backend_factory`).
The first PDF ingestion makes docling download its models (~500 MB), including from
ModelScope; the image does not embed them, so a PDF upload in the container fails with
a generic `422`. No test for any of these.

## LLM04 Data and model poisoning

**Applies to data, not to models.** Verity trains and fine-tunes nothing; the hosted
models' training is outside its control. Data poisoning means the corpus (see LLM08)
and the eval set.

**Mitigation.** Corpus writes are gated as in LLM08 (C23). The eval set and the runs of
record are committed files: a change to them is a reviewed diff, and a stub run cannot
overwrite a live artefact (`tests/unit/test_stub_live_path_separation.py`, C12). Gold
chunk ids must resolve
(`tests/integration/test_eval_labels_integrity.py::test_eval_labels_resolve_to_existing_chunks`
(integration), C07).

**Not stopped.** A key holder's false-but-plausible document (LLM08). No review step
between upload and retrieval, no quarantine, no delete. No test.

## LLM05 Improper output handling

**Applies.** Model output becomes JSON fields, citation offsets and text shown in a
browser.

**Mitigation.** Model output never reaches a shell, an SQL string or a tool: the
sub-questions it writes reach Postgres only as bound parameters
(`PgVectorStore.sparse_search`). A citation is kept only if its quote is a verbatim
substring of the cited chunk; offsets are re-anchored on the source
(`tests/unit/test_synthesizer.py::test_synthesizer_validates_quote_and_reanchors_offsets`,
`tests/unit/test_synthesizer.py::test_synthesizer_drops_hallucinated_citations`, `tests/unit/test_synthesizer.py::test_synthesizer_drops_out_of_range_index`,
`tests/unit/test_citation_verbatim_invariant.py::test_every_citation_quote_is_a_verbatim_substring_of_source_document`,
C01). Malformed scorer output is a refusal
(`tests/unit/test_scorer.py::test_malformed_output_forces_refusal`, C03); malformed
synthesizer output ships nothing
(`tests/unit/test_synthesizer.py::test_synthesizer_never_ships_malformed_output`). The
dashboard escapes every interpolated string with `html.escape`.

**Not stopped.** The web UI renders answer and chunk text through React, which escapes
it, and no component uses `dangerouslySetInnerHTML`; no test asserts that model text is
rendered as text. No test feeds markup to the dashboard. An API consumer other than the
bundled UI must escape the text itself.

## LLM06 Excessive agency

**Does not apply beyond a fixed pipeline.** The model calls no tools and has no write
access: the agent runs decompose → retrieve → synthesize → score in code, and the only
model-controlled input to another component is the list of sub-questions, capped at
`max_agent_steps` (`tests/unit/test_decomposer.py::test_decomposer_caps_at_max_steps`).
The sub-questions only read from the store. Nothing to stop beyond that.

## LLM07 System prompt leakage

**Applies weakly.** The prompts are public files in this repository and in the wheel
(`tests/unit/test_resources.py::test_the_wheel_ships_prompts_and_migrations`, C26); they
hold no credential, and no control depends on their secrecy: the refusal checks are code
that a model which obeys an injection does not bypass
(`tests/unit/test_adversarial.py::test_an_obeyed_injection_does_not_ship_an_invented_answer`).
**Not stopped:** nothing prevents the model from repeating its prompt; no test, by design.

## LLM09 Misinformation

**Applies.** The product's purpose is answers a reader can trust.

**Mitigation.** Verbatim citations (C01); a draft with no surviving citation is refused
(`tests/unit/test_error_vs_refusal.py::test_agent_refuses_an_uncited_draft_without_asking_the_scorer`);
the deterministic support check (C24); threshold-gated refusal
(`tests/unit/test_scorer.py::test_below_threshold_refused_regardless_of_model_flag`,
`tests/unit/test_agent.py::test_agent_refuses_out_of_scope_question`, C02); an outage is
an error, never a refusal or an answer
(`tests/unit/test_error_vs_refusal.py::test_provider_failure_is_an_error_not_a_refusal`, C25).
Measured rates with their intervals: [`SYSTEM_CARD.md`](SYSTEM_CARD.md).

**Not stopped.** A verbatim quote proves the quote, not the sentence around it. The
confidence score is uncalibrated (C21, roadmap); the judge that measures faithfulness is
unvalidated (C27, roadmap); the eval set is small and has no held-out split (C28,
roadmap). A false statement in a corpus document is reported faithfully.

## LLM10 Unbounded consumption

**Applies.** A live `/ask` makes three paid LLM calls, and a PDF upload runs layout and
OCR models.

**Mitigation.**

| Bound | Where | Test |
|---|---|---|
| Key required; 30 requests a minute per key, `/ask` and `/ingest` combined | `api.guards.RateLimiter` | `tests/unit/test_api_hardening.py::test_rate_limit_is_per_key_and_answers_429_with_retry_after` (C23) |
| Question ≤ 2000 characters; body size checked on `Content-Length` before parsing | `api.router.ask_endpoint`, `api.server._guarded` | `tests/unit/test_api_hardening.py::test_question_longer_than_the_limit_is_422` (C23), `tests/unit/test_api_hardening.py::test_oversized_request_body_is_413_before_parsing` |
| Upload ≤ 1 MiB; corpus ≤ 200 documents | `api.router.ingest_endpoint`, `api.state.AppRuntime.ingest_bytes` | `tests/unit/test_api_hardening.py::test_two_megabyte_upload_is_413`, `tests/unit/test_api_hardening.py::test_corpus_document_limit_is_409` (C23) |
| LLM call timeout 30 s, 2 retries per provider | `llm.clients`, `Settings.llm_timeout_s` / `llm_max_retries` | `tests/unit/test_llm_client_resilience.py::test_sdk_clients_get_the_configured_timeout_and_retries` |
| At most `max_agent_steps` sub-questions | `agent.decomposer.LLMQuestionDecomposer` | `tests/unit/test_decomposer.py::test_decomposer_caps_at_max_steps` |
| Output tokens per call: 256 (decompose, score), 512 (synthesize, judge) | call sites in `verity.agent`, `verity.eval.judge` | no test |
| Eval spend cap `verity eval run --budget-usd` (exit 4) | `eval.harness.BudgetExceededError` | no test |
| In-memory spans capped at `VERITY_TRACE_MAX_SPANS` | `observability.tracer` | `tests/unit/test_tracer.py::test_span_buffer_is_bounded_and_keeps_the_newest` (C29) |

**Not stopped.** No per-key or global spending budget on `/ask`, and `verity ask` has
no cap. The limit is per key and per process: k keys on N processes allow k × N × 30
requests a minute. Docling parsing of an accepted PDF has no time limit. No test for any
of these.
