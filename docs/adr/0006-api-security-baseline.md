# ADR 0006 — API security baseline

**Status:** accepted · 2026-11

## Context
Before this decision the API had no authentication, no rate limit, no input bounds and
accepted any file type; a failed ingestion echoed parser internals and temporary file
names to the client. Every `/ask` in live mode makes three paid LLM calls, so an open
endpoint is also an open budget. Verity remains a single-tenant reference
implementation (SECURITY.md), but the baseline below is what any deployment needs
before it leaves a developer's machine.

## Decision
Every limit lives in `Settings` (env prefix `VERITY_`) and is checked in process.

| Control | Default | Where |
|---|---|---|
| `X-API-Key` on `/ask` and `/ingest`; constant-time compare against every key | required | `api/guards.py::require_api_key` |
| Start-up refuses to serve with no key unless `VERITY_API_AUTH_DISABLED=true` | on | `api/server.py::create_app` |
| Sliding-window rate limit per key | 30 / minute, `429` + `Retry-After` | `api/guards.py::RateLimiter` |
| Failed-key limit per client address, checked before the key | 10 / minute, `429` + `Retry-After` | `api/guards.py::require_api_key` |
| Question length | 2000 characters, `422` | `api/router.py` |
| Request body size, checked on `Content-Length` before parsing | upload limit + 16 KiB, `413`; no length `411` | `api/server.py` |
| Upload size | 1 MiB, `413` | `config.py`, `api/server.py` |
| Upload type: extension allow-list, a MIME type consistent with it, content sniffed (PDF/DOCX magic bytes, UTF-8 text without NUL) | `.md .markdown .txt .pdf .docx`, else `415` | `api/guards.py::check_upload` |
| Corpus size | 200 documents, `409` | `api/state.py` |
| Error bodies are generic; detail and traceback go to the structured log under a request ID returned in `X-Request-ID` and the body | always | `api/server.py` |
| Bind address | `127.0.0.1`; other hosts only with an explicit `--host`, with a warning | `cli.py::serve` |

`/health` stays public for load balancers. `/status`, `/metrics`, `/metrics/prom` and
`/dashboard` stay public too: they expose corpus file names and aggregate latency and
cost, no document text. That is accepted for a single tenant and listed as a residual
risk in SECURITY.md.

## Rationale
- **Keys, not users.** One tenant needs to tell its own clients from everyone else, not
  to model users; a key list in the environment is enough and adds no identity store.
- **Fail closed at start-up.** A missing `VERITY_API_KEYS` must not silently produce an
  open API, so disabling auth is a separate, explicit flag.
- **In-process limiter.** One process per tenant means no shared state to coordinate.
  The bucket count is bounded by configuration (keys), not by clients.
- **Bounds from the data.** The longest labelled question is 97 characters and the
  corpus files are 1-4 KB; the limits leave 20x and 250x headroom.

Rejected:
- OAuth / JWT: no identity provider in scope, and it would not change the single-tenant
  threat model.
- A reverse proxy as the only control: the defaults must be safe when someone runs
  `verity serve` directly.
- A Redis-backed limiter: needed only for several API processes, which this deployment
  model does not have.

## Consequences
- Several API processes would each enforce the limit separately; the effective limit
  becomes N × 30 per minute.
- The web UI sends no key: it works against a server started with
  `VERITY_API_AUTH_DISABLED=true` or behind a proxy that adds the header.
- Keys are compared, never logged; logs carry `principal=key-<n>`.
