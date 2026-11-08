# Development

The README's quickstart runs the stub pipeline and the demo. This page has the rest:
the stores, the containers, how the published numbers are reproduced, how a live
evaluation is run, how the project was built, and the checks CI runs.

## Stores and containers

pgvector is the default store (`VERITY_STORE`, or `--store` on the CLI); every command
that reads the corpus first syncs it: unchanged documents are skipped, edited ones have
their chunks replaced, and documents whose file was removed leave the index:

```bash
docker compose up -d db                    # pgvector on :5432
uv run verity db upgrade                   # apply migrations
uv run verity eval retrieval               # ingests datasets/corpus, then scores retrieval
```

`VERITY_STORE=memory` runs everything in process memory, without a database.

Everything in containers (no provider key means stub mode). The image embeds every
local model, so it runs offline, PDF and DOCX uploads included: 4.77 GB uncompressed
on linux/arm64, of which 1.89 GB is the Python environment (torch CPU, docling),
765 MB the docling layout, table and OCR models, 227 MB the embedder and reranker and
204 MB the X11/GL libraries opencv needs.

```bash
VERITY_API_KEYS='["a-long-random-key"]' docker compose up   # or VERITY_API_AUTH_DISABLED=true
curl -H 'X-API-Key: a-long-random-key' -H 'content-type: application/json' \
  -d '{"question": "What is the notice period?"}' http://127.0.0.1:8000/ask
```

## Reproducing the published numbers

The published numbers are recomputed from the committed run artefact, with no model
call:

```bash
uv run python scripts/generate_headline_results.py --check   # scorecard, docs/results.md and README ≡ artefact
```

## Running a live evaluation

A new live evaluation calls paid models (agent and judge): estimate its cost first.
The threshold is chosen on the dev split and the test split is run once at it, with
no change to the system in between
([methodology](evaluation-methodology.md#refusal-deterministic-over-the-agents-decisions)):

```bash
export VERITY_ANTHROPIC_API_KEY=sk-ant-...
uv run verity eval run --judge live --no-agent-stub \
  --dataset datasets/eval/dev.jsonl --budget-usd 2.0 --report-only-floors
uv run python scripts/refusal_analysis.py --dev datasets/eval/runs/live/eval_<sha>_dev.json \
  --out-json scorecards/refusal-analysis.json --out-png docs/figures/reliability.png
VERITY_CONFIDENCE_THRESHOLD=<chosen> uv run verity eval run --judge live --no-agent-stub \
  --dataset datasets/eval/test.jsonl --budget-usd 1.6 --report-only-floors
uv run python scripts/refusal_analysis.py --dev datasets/eval/runs/live/eval_<sha>_dev.json \
  --test datasets/eval/runs/live/eval_<sha>_test.json \
  --out-json scorecards/refusal-analysis.json --out-png docs/figures/reliability.png
uv run python scripts/generate_headline_results.py \
  --artefact datasets/eval/runs/live/eval_<sha>_test.json --scorecard scorecards/<name>.json
```

Run artefacts in `datasets/eval/runs/live/` are pinned in its `SHA256SUMS` and never
edited; the v1.1.0 scorecard comes from the test run
`eval_513b839_test.incomplete-20261107T122042441670Z.json`, and its threshold from the
dev run `eval_6ec544d_dev.json`. The v1.0.0 run (`scorecards/live-v1.0.0.json`) used 16
questions that predate the dev/test split.

## How this was built

Built with coding agents: I write the specs, review the pull requests and their tests, and answer for what gets merged.

The process: a written spec with acceptance criteria, an agent implementing on a
branch, a review of the diff by a second agent without the builder's context, a human
audit of the claims and critical paths, then CI. Standards:
[`docs/ENGINEERING_STANDARDS.md`](ENGINEERING_STANDARDS.md).

Human audits caught what the agents and CI did not:

1. **Machine-specific ids.** Document ids were derived from absolute file paths, so
   the gold chunk ids in the eval dataset only matched on the machine that built
   them. Ids now hash the repo-relative path, and
   `tests/integration/test_eval_labels_integrity.py` checks every gold id resolves.
2. **Scorecard denominators.** The first v1.0.0 headline averaged citation accuracy
   over all 16 questions, counting the 4 refusals as 1.0: 0.792. Restricted to the
   12 answered questions it is 0.722 (v1.0.1). Every answer-quality metric now
   carries its scope and denominator.
3. **An external audit (2026-11-02)** found public claims ahead of the code:
   "calibrated" refusal that was never measured, a methodology defining per-claim
   aggregation while the code averaged per question, a Docker image that cannot
   serve live answers, an API with no input limits or prompt-injection defence,
   pgvector described as the default store, docstrings contradicting the code, and
   sprint-era narrative in shipped files. The README, `docs/CLAIMS.md` and the
   `claims` / `stale-narrative` CI jobs are the first part of the response; the
   [Roadmap](features.md#roadmap) lists what remains.

## Checks

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run vulture && uv run deptry src
HF_HUB_OFFLINE=1 uv run pytest -m "not integration and not slow"  # fast lane, no network
uv run pytest -m "slow and not integration"        # loads the local models
VERITY_DATABASE_URL=postgresql://verity:verity@localhost:5432/verity \
  uv run pytest -m integration                     # needs pgvector
uv run python scripts/check_claims.py              # docs/CLAIMS.md ≡ tests ≡ docs/features.md
uv run python scripts/check_stale_narrative.py
uv run python scripts/generate_headline_results.py --check   # scorecard, docs/results.md and README ≡ artefact
```

[`CLAUDE.md`](../CLAUDE.md) is the working guide for coding agents. See
[CONTRIBUTING.md](../CONTRIBUTING.md), [SECURITY.md](../SECURITY.md), the
[threat model](THREAT_MODEL.md) and the [runbook](RUNBOOK.md). Licensed under
[MIT](../LICENSE).
