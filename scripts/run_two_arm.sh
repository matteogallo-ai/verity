#!/usr/bin/env bash
# Both arms of docs/preregistration/two-arm-comparison.md on dev and test, once each,
# under the caps registered there. Paid: it calls the agent and judge models.
set -euo pipefail

# The artefacts name the commit they ran at; that commit must hold this code.
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "tracked files are modified: commit first" >&2
  exit 1
fi

# Shipped defaults, stated so that a VERITY_ variable in the caller's shell cannot
# change an arm.
export VERITY_STORE=memory VERITY_CONFIDENCE_THRESHOLD=0.0 \
  VERITY_IRRELEVANT_CITATION_CHECK=false VERITY_MIN_CITATION_COVERAGE=0.5
common=(eval run --judge live --no-agent-stub --store memory --report-only-floors)

# Every run goes ahead even if one stops: an incomplete arm is published as is.
status=0
uv run verity "${common[@]}" --dataset datasets/eval/dev.jsonl --budget-usd 1.60 || status=$?
uv run verity "${common[@]}" --dataset datasets/eval/dev.jsonl --budget-usd 2.00 --naive-baseline || status=$?
uv run verity "${common[@]}" --dataset datasets/eval/test.jsonl --budget-usd 2.40 || status=$?
uv run verity "${common[@]}" --dataset datasets/eval/test.jsonl --budget-usd 2.90 --naive-baseline || status=$?
exit "$status"
