# Security policy

## Supported versions

| Version | Supported |
|---|---|
| 1.0.x | yes |
| < 1.0 | no |

Security fixes, including dependency upgrades, go to `main` and ship in the next 1.0.x
patch release. v1.0.1 resolves dependencies with known advisories that `main` has
upgraded (see `CHANGELOG.md`, Unreleased); until a release contains them, build from
`main`.

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
- **Public endpoints.** `/health`, `/status`, `/metrics`, `/metrics/prom` and
  `/dashboard` need no key. They expose corpus file names, document counts and
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

| Threat | Status | Notes |
|---|---|---|
| Indirect prompt injection (instructions inside ingested documents) | **Partial** | Evidence, question and draft answer enter the prompts only inside delimiters, with document text unable to close them, and every prompt says they are untrusted data (`tests/unit/test_prompt_injection.py`). Before the scorer runs, a deterministic check refuses a draft that cites a document carrying an injection marker (verdict keys as JSON, "ignore previous instructions", role lines), whose words its quotes do not cover, or whose citations the reranker scores irrelevant (`verity.agent.support`, `tests/unit/test_support.py`). Adversarial eval questions q-017..q-019 run against a poisoned corpus document with a model that obeys it (`tests/unit/test_adversarial.py`). **Not stopped:** an injection paraphrased past the marker list, and a document that states a false fact relevant to the question. The marker list and the thresholds were tuned on the same 16 questions they are checked on. |
| Corpus poisoning | **Partial** | Upload needs an API key, is size- and type-checked and capped at 200 documents, but any key holder can add a document every other client then retrieves, and with pgvector it persists. Control what is ingested. |
| Cost / denial of wallet | **Partial** | `/ask` needs a key and is rate-limited per key; a live answer makes 3 LLM calls under SDK timeouts and bounded retries. No per-key spending budget. `verity eval run --budget-usd` caps an eval run; `verity ask` has no cap. |
| Information disclosure | **Partial** | Answers quote only ingested documents and errors are generic. Every key holder can query every document; the public read-only endpoints list corpus file names. |
| Secret leakage | **Mitigated** | Provider keys and API keys are read from the environment only and held as `SecretStr`, so `repr()` / `str()` / `model_dump()` of the settings do not show them (`tests/unit/test_config.py::test_api_keys_never_appear_in_settings_repr`); API keys are never logged. The raw provider key is passed to the provider SDK client. |
| Fabricated citations | **Mitigated** | A citation is kept only if its quote is a verbatim substring of the cited chunk (`tests/unit/test_citation_verbatim_invariant.py`, claim C01); an answer left without a citation is refused (`no_citation`). This does not prove the answer text is faithful. |
| Third-party model download | **Partial** | The Docker image embeds the embedding and reranker models. Outside it, the first run downloads them from Hugging Face, and the first PDF ingestion makes docling download its layout and OCR models (~500 MB, including from ModelScope); the image does not embed those, so a PDF upload in the container fails with a generic `422`. |
| Dependency vulnerabilities | **Partial** | Dependencies are pinned in `uv.lock` and `web/pnpm-lock.yaml`, and GitHub Actions by commit SHA. Dependabot opens one grouped update PR per week for uv, the `web/` npm packages and the actions. The `osv-scanner` CI job fails on PRs to `main` and pushes to `main` when the lockfiles resolve a package with a known vulnerability in the OSV database; the accepted exceptions, each with its reason and an expiry date, are in `osv-scanner.toml` (an unfixed `accelerate` advisory whose vulnerable functions nothing calls, and a dev-only `minimatch` advisory). CodeQL analyses the Python and TypeScript source. **Not covered:** a vulnerability not yet in the OSV database, the `python:3.12-slim` base image, and the models downloaded at run time (see the row above). Details in [`docs/security/README.md`](docs/security/README.md). |

This document is updated in the same pull request as any change to the posture
above.
