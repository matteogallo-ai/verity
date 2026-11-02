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

Verity is a **single-tenant reference implementation**: one API process serves one
team's corpus. Run it on your machine, on a trusted network, or behind your own
gateway. The controls below make the defaults safe to run; they are not a multi-tenant
isolation model.

## Current posture

Stated as the code behaves, each with the test that pins it. The design and the
rejected alternatives are in [ADR 0006](docs/adr/0006-api-security-baseline.md).

- **Authentication.** `/ask` and `/ingest` require `X-API-Key`; keys come from
  `VERITY_API_KEYS` (held as `SecretStr`, compared in constant time, never logged; logs
  carry `principal=key-<n>`). With no key configured the server refuses to start unless
  `VERITY_API_AUTH_DISABLED=true` is set, which is for local development only
  (`tests/unit/test_api_hardening.py`).
- **Public endpoints.** `/health`, `/status`, `/metrics`, `/metrics/prom`,
  `/dashboard`, `/docs` and `/openapi.json` need no key. They expose corpus file names, document counts and
  aggregate latency and cost, never document text.
- **Rate limit.** 30 requests per minute per key on `/ask` and `/ingest`, sliding
  window, `429` with `Retry-After`. In process: N API processes allow N × the limit.
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
  never a refusal.
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
  relevant document, is not stopped (LLM01, LLM09). The eval judge prompt has no
  untrusted-content delimiters (LLM01).
- The Postgres DSN, password included, is a plain string in the settings; questions and
  retrieved passages go to the LLM provider (LLM02).
- No per-key spending budget on `/ask` (LLM10); no automated dependency vulnerability
  scan (LLM03).

This document and the threat model are updated in the same pull request as any change
to the posture above.
