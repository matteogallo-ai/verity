# Security policy

## Supported versions

| Version | Supported |
|---|---|
| 1.1.x | yes |
| < 1.1 | no |

Security fixes, including dependency upgrades, go to `main` and ship in the next 1.1.x
patch release; until a release contains them, build from `main`.

## Reporting a vulnerability

One channel: GitHub private vulnerability reporting,
[Security → Report a vulnerability](https://github.com/matteogallo-ai/verity/security/advisories/new).
The report reaches the maintainer only. Do not open a public issue. Expect an
acknowledgement within 7 days.

## Intended deployment

Verity is a **single-tenant reference implementation**: one API process serves one
team's corpus. Run it on your machine, on a trusted network, or behind your own
gateway. The controls below make the defaults safe to run; they are not a multi-tenant
isolation model.

## Current posture

Stated as the code behaves, each with the test that pins it. The design and the
rejected alternatives are in [ADR 0006](docs/adr/0006-api-security-baseline.md).

- **Authentication.** `/status`, `/ask` and `/ingest` require `X-API-Key`; keys come from
  `VERITY_API_KEYS` (held as `SecretStr`, compared in constant time, never logged; logs
  carry `principal=key-<n>`). With no key configured the server refuses to start unless
  `VERITY_API_AUTH_DISABLED=true` is set, which is for local development only
  (`tests/unit/test_api_hardening.py`).
- **Public endpoints.** `/health`, `/ready`, `/metrics`, `/metrics/prom`, `/dashboard`,
  `/docs` and `/openapi.json` need no key. They expose readiness states and aggregate
  latency and cost, never document names or text. `/status`, which lists the corpus file
  names and the models in use, needs a key.
- **Failed key attempts.** 10 wrong or missing keys per minute per client address (per
  /64 for IPv6), then `429` with `Retry-After`, checked before the key is compared. The
  limit slows one client; long random keys are what makes guessing infeasible. At most
  10,000 addresses are tracked, so a client with more addresses than that can reset its
  count, and N API processes allow N × the limit. The address is the TCP peer unless
  uvicorn trusts the proxy that forwards it, set with `FORWARDED_ALLOW_IPS=<proxy ip>`
  (`verity serve` passes no value, so uvicorn reads that variable). Under
  `docker compose` with no such setting, every client reaches the container from the
  Docker gateway: the limit is then shared, and one client with a wrong key blocks the
  others for up to a minute.
- **Rate limit.** 30 requests per minute per key on `/status`, `/ask` and `/ingest`, sliding
  window, `429` with `Retry-After`. In process: N API processes allow N × the limit.
- **Spending budget.** `/ask` spend per key is capped at `VERITY_API_BUDGET_USD`
  ($5 by default) over a sliding `VERITY_API_BUDGET_WINDOW_S` (3600 s), counted from
  the cost each answer reports, including answers that fail after paying for some
  calls. Over budget is `429` with `Retry-After` and a generic body. A request is
  admitted while the key is under budget, so the budget can be overshot by the
  answers in flight when it is crossed. In process: N API processes allow
  N × the budget, and a restart resets it (`tests/unit/test_api_hardening.py`).
- **Bounded LLM calls.** Each answer makes at most 3 LLM calls with `max_tokens`
  256, 512 and 256, each under `VERITY_LLM_TIMEOUT_S` (30 s) and
  `VERITY_LLM_MAX_RETRIES` (2) per provider. `verity ask` stops before any call
  whose estimated cost (input bytes as tokens, `max_tokens` as output, the dearest
  configured provider) would take the answer past `--budget-usd`
  (`VERITY_ASK_BUDGET_USD`, $1 by default): exit 4, spend so far printed
  (`tests/unit/test_budget.py`). Bytes bound tokens for OpenAI's tokenizer; for
  Anthropic's, which is not public, it is an assumption.
- **Input bounds.** Questions up to 2000 characters (`422`). Request bodies are checked
  on `Content-Length` before parsing (`411` without one, `413` over the limit); uploads
  up to 1 MiB (`413`).
- **Upload types.** `.md`, `.markdown`, `.txt`, `.pdf`, `.docx` only. The MIME type
  must match the extension, and the content is sniffed (PDF and DOCX magic bytes,
  UTF-8 text without NUL bytes); anything else is `415`.
- **Corpus cap.** 200 documents (`409`). With pgvector (the default) uploads persist
  across restarts; no endpoint deletes a document.
- **Errors.** Client errors are generic. Parser messages, stack traces and temporary
  file names go to the structured log under a request ID, returned in the
  `X-Request-ID` header and the error body. An unreachable store or provider is `503`,
  never a refusal; a defect in the code is a generic `500`, and the calls it already paid
  for count against the key's spending budget.
- **Network exposure.** `verity serve` binds `127.0.0.1`; another host needs an
  explicit `--host` and prints a warning. Docker Compose runs the API on `0.0.0.0`
  inside the container and publishes it, and Postgres, on the host's `127.0.0.1`
  only. CORS allows `http://localhost:3000`; CORS does not stop non-browser clients.
- **Container.** The runtime image runs as an unprivileged user (uid 10001), holds no
  compiler or source tree, and loads the embedded models with `HF_HUB_OFFLINE=1`.
- **Development credentials.** Docker Compose uses fixed Postgres credentials
  (`verity` / `verity`); change them for anything shared.
- **Web UI.** It sends no API key: use it against a server started with
  `VERITY_API_AUTH_DISABLED=true` or behind a proxy that adds the header.

## Threat model

[`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md) goes through the OWASP Top 10 for LLM
Applications 2025 item by item: whether each applies, the code that mitigates it, the
tests that prove it, and what is not stopped. The residual risks that matter most for a
deployment:

- Every key holder reads every document, and can add one that every other client then
  retrieves, persistently on pgvector; no upload is logged with its key (LLM08, LLM04).
- An injection paraphrased past the marker list, or a false fact stated plainly in a
  relevant document, is not stopped (LLM01, LLM09). The published answer-quality
  figures were judged by eval judge prompt v0.1.0, which has no untrusted-content
  delimiters; v0.2.0 has them (LLM01).
- Questions and retrieved passages go to the LLM provider (LLM02).
- The `/ask` spending budget is per process and resets on restart, and failed attempts
  a provider bills are not counted (LLM10). The dependency scan misses advisories not
  yet in the OSV database, the base image and the models downloaded at run time
  (LLM03; [`docs/security/README.md`](docs/security/README.md)).

This document and the threat model are updated in the same pull request as any change
to the posture above.
