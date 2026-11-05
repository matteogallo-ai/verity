#!/usr/bin/env bash
# Both arms of docs/preregistration/naive-rag-baseline.md, once each, under the caps the
# owner approved. Paid: it calls the agent and judge models.
set -euo pipefail

VERITY_CAP_USD=0.92
NAIVE_CAP_USD=1.22

# The artefacts name the commit they ran at; that commit must hold this code.
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "tracked files are modified: commit first" >&2
  exit 1
fi

export VERITY_STORE=memory VERITY_CONFIDENCE_THRESHOLD=0.0
common=(eval run --judge live --no-agent-stub --dataset datasets/eval/dev.jsonl
        --store memory --report-only-floors)

# The second arm runs even if the first stops: an incomplete arm is published as is.
status=0
uv run verity "${common[@]}" --budget-usd "$VERITY_CAP_USD" || status=$?
uv run verity "${common[@]}" --budget-usd "$NAIVE_CAP_USD" --naive-baseline || status=$?
exit "$status"
