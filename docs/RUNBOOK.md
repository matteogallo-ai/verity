# Runbook

The failures most likely to reach an operator of `verity serve` (or the Compose stack)
and of `verity eval run`, each with the symptom, the signal the code actually emits,
and the response. Log events are structlog lines (JSON keys `event`, `trace_id`,
`request_id`, `principal`); every error body carries the `request_id` that finds them.

## Signals available

| Signal | What it tells | What it does not |
|---|---|---|
| `GET /health` | liveness: the process answers HTTP | nothing about Postgres, the models or the providers: it returns `200` while all three are down, so a restart policy keyed on it never restarts a healthy process for a dependency outage |
| `GET /ready` | readiness: `200` when the startup preload is done (every corpus file ingested, embedding and reranker models loaded), Postgres answers `SELECT 1` through the store's connection path (if it is the configured store; the first ping also checks the schema, so an unmigrated database or a wrong embedding dimension reads `unreachable` too) and the LLM providers can serve: they are probed in routing order with a Models API lookup (no tokens), moving on only when one is unavailable, so a rejected key or an unknown model on the first provider is `503` even if the next one answers. Otherwise `503`. Body `{"status": "ready" \| "not ready", "checks": {"store": …, "llm": …, "preload": …}}`; `store` and `llm` are `ok`, `unreachable`, `timeout` (2 s per check) or `not used` (memory store, stub agent), `preload` is `loading`, `ok` or `failed`; the cause is in the `readiness_failed` / `readiness_timeout` / `preload_failed` log lines, never in the body. A dependency result is reused for 5 s, so a caller looping on this public endpoint reaches the dependencies at most once per 5 s; the preload state is read on every call, and a failed preload is started again | whether the provider still has rate limit or credit left for completions; a recovery shows up to 5 s late |
| `GET /status` | the store answers (it lists the corpus) and the boot-time provenance; `provenance.is_stub: true` means no provider key was found and answers are canned | provider health |
| `answer_done` log line | one per answered or refused request: `decision`, `reason`, `stage_ms`, `latency_ms`, tokens, `cost_usd` | — |
| `answer_failed` log line | one per request that could not be decided: `decision=error`, `error_kind`, `error`, the stages reached | tokens spent before the failure |
| `/metrics`, `/metrics/prom` | per-stage latency percentiles, refusal count and rate, session cost | errors: no counter exists; count `answer_failed` lines or `503` responses |

`error_kind` values: `provider_unavailable`, `provider_auth`, `provider_error`,
`retrieval_unavailable` (`verity.types.ErrorKind`). A refusal is never an error and an
error is never a refusal (C25).

## 1. LLM provider down, rate-limited (429) or timing out

- **Symptom.** `/ask` answers `503` with `{"detail": {"error": "provider_unavailable", …}}`
  after a long wait: each provider gets 30 s × 3 attempts (`llm_timeout_s`,
  `llm_max_retries`) before the router moves on.
- **Detection.** A `provider_unavailable_fallback` warning per provider that failed
  (`verity.llm.routing`), then `answer_failed` with `error_kind=provider_unavailable`.
  Timeouts, connection errors, `429` and `5xx` all map here
  (`verity.llm.clients._raise_translated`). Tests:
  `tests/unit/test_routing.py::test_falls_back_on_provider_unavailable`,
  `tests/unit/test_api_router.py::test_ask_answers_503_when_every_provider_is_down`.
- **Response.** Check the provider's status page. Fallback needs the second provider's
  key (`VERITY_OPENAI_API_KEY`); without it the second attempt fails the same way. On
  sustained `429`, lower `VERITY_API_RATE_LIMIT_PER_MINUTE` or the number of keys.
  Advise API clients to retry a `503` with backoff (the response carries no
  `Retry-After`); the request id ties a client report to the log.
- **Variant: bad key.** `auth_error_short_circuit` then `answer_failed` with
  `error_kind=provider_auth`; no fallback by design
  (`tests/unit/test_routing.py::test_authentication_error_short_circuits`). Fix the key
  and **restart**: settings are read once per process (`verity.config.get_settings`).

## 2. Postgres unreachable or not migrated

- **Symptom.** `/ask`, `/ingest` and `/status` answer `503` with
  `{"detail": {"error": "retrieval_unavailable", …}}`; `/health` stays `200`;
  `/ready` answers `503` with `"store": "unreachable"` (or `"timeout"`).
- **Detection.** `store_unavailable` (error, with traceback) when the failure is raised
  outside the agent, e.g. during a request's corpus ingestion; `stage_failed`
  then `answer_failed` with `error_kind=retrieval_unavailable` inside `/ask`. Tests:
  `tests/unit/test_api_hardening.py::test_unreachable_postgres_is_a_generic_503`,
  `tests/unit/test_api_router.py::test_ask_answers_503_when_the_store_is_down`.
- **Response.** `docker compose ps db`; check `VERITY_DATABASE_URL`. A `store_unavailable`
  traceback naming `SchemaNotMigratedError` needs `uv run verity db upgrade`; one naming
  `EmbeddingDimensionMismatchError` means `VERITY_EMBEDDING_DIM` or the embedding model
  changed without a migration and a re-embed. No restart is needed once Postgres is
  back: every call opens a new connection.

## 3. Model cache missing with `HF_HUB_OFFLINE=1`

Outside the Docker image (which embeds the embedding and reranker models), a fresh
machine or a cleared Hugging Face cache with network access disabled.

- **Symptom.** `/ready` stays `503` with `"preload": "failed"` (or `"loading"` while an
  attempt runs); `/health` stays `200`. `/ask` answers a generic `500`. `/ingest` answers
  `422` "The document could not be parsed." for any file: the error is blamed on the
  document. The corpus is not ingested, and `/status` lists only what Postgres already
  held.
- **Detection.** At startup, one `embed_failed` line per corpus file, with `error`
  reading "We couldn't connect to 'https://huggingface.co' … couldn't find them in the
  cached files" (an `OSError` from the model loader, reproduced with an empty
  `HF_HOME`), then `preload_failed` with the number of failed files. Then
  `unhandled_error` with that traceback on each `/ask`, and `ingest_failed` on each
  upload. The not-ready state and the retry are tested with a loader that fails once
  (`tests/unit/test_readiness.py::test_a_failed_preload_is_not_ready_and_is_retried`);
  the missing cache itself is not.
- **Response.** Fill the cache once with network access (a command that loads both
  models, e.g. `uv run verity retrieve "warm up" --store memory`), or run the Docker
  image. No restart is needed: after a failed preload, `/ready` starts another attempt
  (at most one per 5 s; each embeds the files not yet indexed; requests such as
  `/status` never start one), and the process turns ready
  once every corpus file is in.

## 4. Budget exhausted

- **Eval run cap.** `verity eval run --budget-usd X` stops after the question whose
  cumulative agent + judge cost exceeds `X` (`eval.harness.BudgetExceededError`): the
  console prints "Aborted after q-…: cumulative cost $… exceeded budget cap $…", then the
  path of the incomplete run and the total spend, and the command exits `4`. Errored
  examples count against the cap. The cap is checked after each question, so the
  overshoot is up to one question's cost
  (`tests/unit/test_incomplete_runs.py::test_a_budget_abort_carries_every_example_paid_so_far`).
  Response: re-estimate the cost (question count × the per-answer cost in
  [`SYSTEM_CARD.md`](SYSTEM_CARD.md), plus the judge) and rerun with an approved cap.
- **Incomplete runs.** A run stopped by the cap, or with errored examples (agent or judge
  outage, exit `6`), is written to
  `eval_<sha>_<dataset>.incomplete-<UTC time>.json` next to the runs of record. It never
  overwrites a file, `verity eval compare` ignores it, and no headline scorecard is
  written for it. It holds every paid example: scored ones in `per_audit`, the others in
  `errored` with the failed stage, the error kind, their usage and stage timings, and,
  when the judge failed, the answer and its citations, so they can be judged again
  without paying the agent twice
  (`tests/unit/test_incomplete_runs.py::test_an_incomplete_run_never_replaces_or_joins_the_runs_of_record`).
  Not covered: a rejected judge key (the run stops at once, as every example would fail
  alike), an exception that is not a provider error, or a failure to write the
  incomplete run itself; these end the run with a traceback and no artefact.
- **Provider account out of credit or quota.** Verity has no notion of it; it sees the
  provider's HTTP error. A `429` is treated as unavailability (section 1); any other
  `4xx` becomes `answer_failed` with `error_kind=provider_error` and is not sent to the
  fallback provider: the router catches only authentication and unavailability errors
  (`verity.llm.routing`; no routing test covers this case). The status-to-error mapping
  is tested in `tests/unit/test_llm_client_resilience.py::test_sdk_errors_translate_by_class`.
  The `error` field of the log line holds the provider's message. Response: top up or
  raise the quota; no restart needed.
- **Per-key API budget.** A key whose `/ask` spend reaches `VERITY_API_BUDGET_USD`
  within `VERITY_API_BUDGET_WINDOW_S` gets `429` with `Retry-After` until enough spend
  leaves the window
  (`tests/unit/test_api_hardening.py::test_spending_budget_is_per_key_and_answers_429_with_retry_after`).
  The budget is per process and a restart resets it, so it does not cap total spend:
  watch `verity_cost_usd_total` and the `cost_usd` of `answer_done`, and lower the
  budget or revoke keys (remove from `VERITY_API_KEYS`, restart).
- **`verity ask` cap.** `--budget-usd` (`VERITY_ASK_BUDGET_USD`) stops before a call
  whose estimated upper cost could cross it: exit `4`, spend so far printed
  (`tests/unit/test_budget.py::test_ask_stops_at_the_budget_with_exit_4_and_prints_the_spend`).

## 5. Malformed or oversized upload

- **Symptom and detection.** Status codes from `POST /ingest`, each with a generic body
  and the request id:

  | Status | Cause | Test |
  |---|---|---|
  | `411` | no `Content-Length` | no test |
  | `413` | body or file over 1 MiB | `tests/unit/test_api_hardening.py::test_two_megabyte_upload_is_413` |
  | `415` | extension, MIME type or content outside the allow-list | `tests/unit/test_api_hardening.py::test_disallowed_upload_type_is_415` |
  | `400` | empty file | `tests/unit/test_api_router.py::test_ingest_rejects_empty_upload` |
  | `401` | missing or wrong `X-API-Key` | `tests/unit/test_api_hardening.py::test_missing_key_is_401` |
  | `429` | 30 requests a minute per key, `/ask` and `/ingest` combined | `tests/unit/test_api_hardening.py::test_rate_limit_is_per_key_and_answers_429_with_retry_after` |
  | `409` | corpus at 200 documents | `tests/unit/test_api_hardening.py::test_corpus_document_limit_is_409` |
  | `422` | parser failure; `ingest_failed` warning with the traceback | `tests/unit/test_api_hardening.py::test_corrupt_pdf_is_a_generic_422_with_a_request_id` |

- **Response.** Client-side for `411`/`413`/`415`/`400`. For `422`, read `ingest_failed`
  under the request id: a corrupt file is the client's; an `OSError` from a model loader
  is section 3; a PDF in the container fails because docling's models are not in the
  image. A `409` needs documents removed in SQL (below); no endpoint deletes one.

## 6. A poisoned or injection document in the corpus

- **Symptom.** Questions on a topic are refused that used to be answered, or answered
  with a false fact.
- **Detection.** `answer_done` with `decision=refused` and `reason=suspected_injection`
  (marker found in a cited document), `low_citation_coverage` or `irrelevant_citation`;
  `verity_refusal_rate` rising. The `/ask` response's `hits_used` lists the
  `document_id` of each retrieved chunk. A plainly false document raises no signal.
  Tests: `tests/unit/test_support.py::test_a_marker_anywhere_in_the_document_distrusts_every_cited_chunk`,
  `tests/unit/test_adversarial.py::test_an_obeyed_injection_does_not_ship_an_invented_answer`.
- **Response.** Find the document, e.g.
  `SELECT DISTINCT d.id, d.uri, d.title FROM chunks c JOIN documents d ON d.id = c.document_id WHERE c.text ~* 'ignore (all |any )?(the )?(previous|prior|above) instructions';`
  then `DELETE FROM documents WHERE id = '<id>';` (chunks cascade). Uploads are not
  logged with their key, so the uploader cannot be identified from the logs; rotate keys
  if the source is unknown. With `VERITY_STORE=memory`, restart the process.
