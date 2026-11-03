#!/usr/bin/env bash
# Smoke test of the Compose stack: build, bring up db → migrate → ingest → app, then
# check /health, a non-empty corpus on /status (with the key), 401 without a key and one
# stub /ask.
# Used by the CI `docker` job; runnable locally. No provider key: the agent is the stub.
set -euo pipefail

export VERITY_API_KEYS='["docker-smoke-key-0123456789"]'
export VERITY_API_PORT="${VERITY_API_PORT:-8000}"
project="${COMPOSE_PROJECT_NAME:-verity-smoke}"
base="http://127.0.0.1:${VERITY_API_PORT}"
key="docker-smoke-key-0123456789"

cleanup() {
  docker compose -p "$project" logs --no-color migrate ingest app | tail -n 60 || true
  docker compose -p "$project" down -v
}
trap cleanup EXIT

docker compose -p "$project" up --build -d --wait

curl -fsS "$base/health"
echo

curl -fsS "$base/status" -H "X-API-Key: $key" | python3 -c '
import json, sys
corpus = json.load(sys.stdin)["corpus"]
assert corpus["n_documents"] > 0 and corpus["n_chunks"] > 0, corpus
print("status:", corpus["n_documents"], "documents,", corpus["n_chunks"], "chunks")
'

code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$base/ask" \
  -H 'content-type: application/json' -d '{"question": "x"}')
[ "$code" = 401 ] || { echo "expected 401 without a key, got $code"; exit 1; }
echo "ask without key: 401"

curl -fsS -X POST "$base/ask" -H "X-API-Key: $key" -H 'content-type: application/json' \
  -d '{"question": "What is the notice period for terminating the agreement?"}' \
  | python3 -c '
import json, sys
answer = json.load(sys.stdin)["answer"]
assert not answer["confidence"]["refused"] and answer["citations"], answer
print("ask: answered with", len(answer["citations"]), "citation(s):", answer["text"][:80])
'
