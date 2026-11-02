# CLAUDE.md — Verity

Read this first. It is enough to work in this repo; `docs/ENGINEERING_STANDARDS.md` is
the full standard, `docs/CLAIMS.md` the list of what the README promises.

## What Verity is

1. A RAG agent over enterprise documents: decompose → hybrid retrieve → cited synthesis → confidence score.
2. It refuses ("I don't know") when the scorer refuses or its score is below `confidence_threshold`.
3. Every citation quote is a verbatim substring of its source document; others are dropped.
4. An eval harness measures retrieval, refusal and LLM-judged answer quality; one live run is the published artefact.
5. Surfaces: `verity` CLI, FastAPI app (`verity serve`), Next.js UI in `web/`. Python 3.12, uv.

## Layout

```
src/verity/
  types.py            frozen domain model shared by every stage
  config.py           Settings (env prefix VERITY_), the only place for knobs
  cli.py              Typer CLI: version, ingest, db upgrade, ask, retrieve, serve, eval {retrieval,run,compare}
  runtime.py          composition shared by CLI and API (store, embedder, reranker, agent)
  ingestion/          parsers (markdown, text, docling, web), chunker, embedder, ids, pipeline
  retrieval/          memory_store (cosine+BM25), pgvector_store, fusion (RRF), reranker, hybrid
  agent/              agent, decomposer, synthesizer, scorer, evidence, scripted_stub
  llm/                clients (Anthropic, OpenAI, stub), routing (fallback), prompt loader, json_output
  eval/               retrieval_metric, refusal, judge, harness, headline (scorecard v3), run_store
  observability/      tracer (OTel in-memory), logging (structlog), warmup
  api/                server (/health, /metrics, /dashboard), router (/status, /ask, /ingest), state
  db/                 SQL migration runner
prompts/              *.prompt files loaded by verity.llm.prompt
migrations/           numbered SQL files (pgvector schema)
datasets/corpus/      21 documents (Markdown, PDF, DOCX; one adversarial), see datasets/CORPUS.md
datasets/corpus_src/  Markdown sources of the PDF/DOCX documents
datasets/eval/        questions.jsonl (95), frozen dev.jsonl / test.jsonl split
scorecards/           committed live scorecard (source of the README table)
scripts/              check_claims, check_stale_narrative, check_commit_messages,
                      generate_headline_results, build_eval_labels,
                      build_corpus_documents, split_dataset
tests/unit/           fast + slow lanes; tests/integration/ needs Postgres
web/                  Next.js 15 UI (pnpm, vitest)
```

## Commands

```bash
uv sync --dev                                              # install (Python 3.12, CPU torch on Linux)
uv run ruff check . && uv run ruff format --check .        # lint + format
uv run mypy                                                # typecheck (strict)
uv run vulture                                             # dead code
uv run deptry src                                          # dependency hygiene
HF_HUB_OFFLINE=1 uv run pytest -m "not integration and not slow"  # fast lane: < 60 s, network blocked
uv run pytest -m "slow and not integration"                # slow lane: loads local models (~3 min)
docker compose up -d db && uv run verity db upgrade        # pgvector for the integration lane
uv run pytest -m integration                               # integration lane (VERITY_DATABASE_URL)
uv run python scripts/check_claims.py                      # CLAIMS.md ≡ passing tests ≡ README (needs Postgres)
uv run python scripts/check_claims.py --no-run             # structure only, no tests executed
uv run python scripts/check_stale_narrative.py             # no sprint labels / future tense / bare TODO
uv run python scripts/check_commit_messages.py main..HEAD  # Conventional Commits on the branch
uv run python scripts/generate_headline_results.py --check # README results block ≡ scorecard
uv run python scripts/build_corpus_documents.py --check    # PDF/DOCX corpus ≡ sources
uv run python scripts/split_dataset.py --check             # dev/test split ≡ seed
uv run python scripts/recompute_live_aggregates.py --check # live runs: *.recomputed.json current
uv run python scripts/refusal_analysis.py --check --dev datasets/eval/runs/live/eval_6ec544d_dev.json \
  --test datasets/eval/runs/live/eval_969e1d5_test.json --out-json scorecards/refusal-analysis.json
uv run python -m tests.docling_fixture                     # refresh the parsed PDF/DOCX fixture of the fast lane
VERITY_STORE=memory uv run verity eval run --ci            # pipeline smoke test on stubs (no keys, no db)
VERITY_API_AUTH_DISABLED=true uv run verity serve --port 8000  # API on pgvector, dev: no auth, stub without LLM key
make demo                                                  # pgvector + API + web UI, stub mode
VERITY_API_AUTH_DISABLED=true docker compose up            # db → migrate → ingest → app on 127.0.0.1:8000 (C26)
scripts/docker_smoke.sh                                    # build + compose + /health /status /ask (CI job docker)
cd web && corepack pnpm lint && corepack pnpm typecheck && corepack pnpm test
```

## Repo-specific rules

- `make demo`, CI and the tests never call a paid model. Any live run (`--judge live`,
  `--no-agent-stub`, `verity ask` without `--stub`) needs a cost estimate and approval.
- The README results block is generated; edit `scorecards/` or the generator, never the block.
- Changing a non-comment line of a `prompts/*.prompt` file changes model behaviour.
- `datasets/eval/runs/live/` holds runs of record, pinned in its `SHA256SUMS`; never
  modify them, even a derived aggregate. Recomputations go in `*.recomputed.json`.
- A new README feature needs a `docs/CLAIMS.md` row with passing tests in the same PR.

## Appendix A — How coding agents must work in my repos

Every repo's `CLAUDE.md` reproduces this appendix verbatim, below a short repo-specific
section (what it is, layout, exact commands). Its purpose: make the agent engineer, not
generate.

### Engineer, don't generate
1. **Read before you write.** Find the existing module, contract and test for what you are
   changing. Extend it; never create a parallel version.
2. **Smallest change that meets the acceptance criteria.** No speculative abstractions, no
   "for future use" parameters, no options nobody asked for.
3. **Second copy = extract.** About to paste a helper? Move it to a shared module and test it once.
4. **Failing test first** for every bug: show it red, then fix, then show it green.
5. **Name the failure modes** of what you write (timeout, empty input, malformed model output,
   provider down, oversized input, hostile content) and test the ones that matter.
6. **Comments explain why — never what, never when.** No sprint IDs, no "will be added later",
   no restating the code.
7. **Deterministic code over model calls** whenever the answer is computable. The model
   proposes; code validates.
8. **Never report success without the command output that proves it.**
9. **Challenge with evidence.** If an instruction, finding or ADR looks wrong, say so with file,
   line and output — do not comply silently, do not "fix" something that is not broken.
10. **Explain trade-offs** in the PR: what you chose, what you rejected, why.
11. **Independent review before handing back**: a fresh-context reviewer subagent checks the
    diff against these standards; report what it found, what you fixed, what you rejected and why.

### Forbidden
- Claiming in README/docs a feature that has no passing test listed in `docs/CLAIMS.md`.
- Future tense or sprint references in source code or docs.
- Presenting any score computed on a stub, mock or fixture as a quality measure.
- Concatenating untrusted content (retrieved, ingested, scraped, tool output) into a prompt
  without delimiters and an explicit untrusted-content instruction.
- Catch-all exception handlers without a justifying comment and a typed re-raise or log.
- Secrets held in plain strings where the framework offers a secret type
  (e.g. pydantic `SecretStr`).
- Any paid model call without a cost estimate and explicit approval.
- Commits over ~400 changed lines (lockfiles and generated artefacts excluded).
- Pushing, tagging or publishing: the human does that.

### Report format (end of every work unit, in French for Matteo)
1. What changed, point by point. 2. Evidence: commands + outputs. 3. Audit findings disproved,
with proof. 4. Reviewer findings: fixed / rejected + why. 5. Open items. 6. Walkthrough per
changed module — what it does, why this design, how it is verified, how it breaks — written
for someone who pilots agents well but did not hand-write the code. 7. Five comprehension
questions Matteo must answer before merging (expected answers folded at the end).
