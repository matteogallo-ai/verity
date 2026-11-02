# Security policy

## Supported versions

| Version | Supported |
|---|---|
| 1.0.x | yes |
| < 1.0 | no |

## Reporting a vulnerability

Use GitHub's private reporting:
[Security Advisories → Report a vulnerability](https://github.com/matteogallo-ai/verity/security/advisories/new).
Do not open a public issue. Expect an acknowledgement within 7 days.

## Intended deployment

Verity 1.0.x is a **single-tenant reference implementation for local use or a trusted
network. It is not hardened for exposure to the Internet.** Run it on your machine
or behind your own authenticating gateway, with documents you trust.

## Current posture (1.0.x)

Stated as the code behaves today, verified against the source:

- **No authentication and no rate limit** on any endpoint (`/ask`, `/ingest`,
  `/status`, `/metrics`, `/dashboard`).
- **`verity serve` binds `0.0.0.0` by default**, so the API is reachable from the
  local network unless `--host 127.0.0.1` is passed. CORS allows
  `http://localhost:3000` only; CORS does not protect against non-browser clients.
- **Input sizes are not bounded.** `/ask` requires a non-empty question with no
  maximum length; `/ingest` reads the whole upload into memory with no size limit.
- **`/ingest` accepts any file type** (unknown extensions are parsed as text) into a
  **corpus shared by every client**, with no count limit and no deletion endpoint.
  The corpus lives in process memory and is reset on restart.
- **Some parsing errors reach the client.** A failed ingestion returns
  `422 Ingestion failed: <exception text>`, which can include parser internals and
  temporary file names.
- **Docker Compose uses fixed database credentials** (`verity` / `verity`) and
  publishes Postgres on port 5432; it is a development setup.

## Threat model

| Threat | Status | Notes |
|---|---|---|
| Indirect prompt injection (instructions inside ingested documents) | **Not mitigated** | Retrieved chunk text is inserted into the synthesis and confidence prompts without delimiters or an untrusted-content instruction. A document can try to steer the answer or the refusal decision. |
| Cost / denial of wallet | **Not mitigated** on the API | Every `/ask` in live mode makes 3 LLM calls, with no auth, rate limit or budget. The CLI's `verity eval run --budget-usd` caps an eval run; `verity ask` has no cap. |
| Corpus poisoning | **Not mitigated** | Any client can upload into the shared corpus; uploads cannot be removed without a restart. |
| Information disclosure | **Partial** | Answers only quote ingested documents, but every client can query every uploaded document, and some error messages expose internals (see above). |
| Secret leakage | **Partial** | Provider keys are read from the environment only (`VERITY_ANTHROPIC_API_KEY`, `VERITY_OPENAI_API_KEY`) and held as pydantic `SecretStr`, so `repr()` / `str()` / `model_dump()` of the settings do not show them (`tests/unit/test_config.py::test_api_keys_never_appear_in_settings_repr`). No code path logs them (checked by grep over `src/`). The raw key is passed to the provider SDK client. |
| Fabricated citations | **Mitigated** | A citation is kept only if its quote is a verbatim substring of the cited chunk; offsets are re-anchored to the source document (`tests/unit/test_citation_verbatim_invariant.py`, claim C01). This does not prove the answer text is faithful. |
| Dependency vulnerabilities | **Partial** | Dependencies are pinned in `uv.lock` and `web/pnpm-lock.yaml`; no automated vulnerability scan runs. |

This document is updated in the same pull request as any change to the posture
above.
