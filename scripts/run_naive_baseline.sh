#!/usr/bin/env bash
# Both arms of docs/preregistration/naive-rag-baseline.md, once each, under the caps the
# owner approved. Paid: it calls the agent and judge models.
#
#   scripts/run_naive_baseline.sh              # the runs of record (commit e644a98)
#   scripts/run_naive_baseline.sh completion   # amendment 2026-11-05: the questions
#                                              # each arm did not complete
set -euo pipefail

case "${1:-}" in
  "")
    VERITY_CAP_USD=0.92
    NAIVE_CAP_USD=1.22
    verity_dataset=datasets/eval/dev.jsonl
    naive_dataset=datasets/eval/dev.jsonl
    extra=()
    ;;
  completion)
    # The harness checks its cap after each question, so an arm can pass it by one
    # question ($0.087 and $0.075 at most in the e644a98 runs, an observation, not a
    # limit). The warmup is a paid answer outside the cap, and latency is not a
    # registered metric, so it is off.
    VERITY_CAP_USD=0.42
    NAIVE_CAP_USD=0.48
    verity_dataset=datasets/eval/dev-completion.jsonl
    naive_dataset=datasets/eval/dev-completion-11.jsonl
    extra=(--no-warmup)
    uv run python scripts/build_completion_subset.py --check
    # The completed rows ran at e644a98; the new rows join them only if the code and
    # the locked dependencies (the reranker's scores depend on them) are the same.
    git cat-file -e 'e644a98^{commit}' 2>/dev/null \
      || { echo "commit e644a98 is not in this clone" >&2; exit 1; }
    git diff --quiet e644a98 HEAD -- src prompts migrations datasets/corpus pyproject.toml uv.lock \
      || { echo "pipeline code differs from e644a98; amend first" >&2; exit 1; }
    # Settings also reads .env and the environment: only keys may be set there.
    overrides="$( (env; [ -f .env ] && cat .env) | grep -E '^VERITY_' \
      | grep -vE '^VERITY_(ANTHROPIC|OPENAI)_API_KEY=' || true)"
    [ -z "$overrides" ] || { echo "VERITY_ settings set outside the registration: ${overrides%%=*}" >&2; exit 1; }
    ;;
  *) echo "usage: $0 [completion]" >&2; exit 2 ;;
esac

# The artefacts name the commit they ran at; that commit must hold this code.
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "tracked files are modified: commit first" >&2
  exit 1
fi

export VERITY_STORE=memory VERITY_CONFIDENCE_THRESHOLD=0.0
common=(eval run --judge live --no-agent-stub --store memory --report-only-floors)

# The second arm runs even if the first stops: an incomplete arm is published as is.
status=0
uv run verity "${common[@]}" ${extra[@]+"${extra[@]}"} --dataset "$verity_dataset" \
  --budget-usd "$VERITY_CAP_USD" || status=$?
uv run verity "${common[@]}" ${extra[@]+"${extra[@]}"} --dataset "$naive_dataset" \
  --budget-usd "$NAIVE_CAP_USD" --naive-baseline || status=$?
exit "$status"
