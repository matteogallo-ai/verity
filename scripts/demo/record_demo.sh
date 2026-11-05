#!/usr/bin/env bash
# Record the demo: Compose stack (db → migrate → ingest → API) on the shipped corpus,
# the web UI built against it, then scripts/demo/record.py drives the UI in Chromium
# and cuts docs/media/verity-demo.{gif,mp4}.
#
#   scripts/demo/record_demo.sh stub   # $0: scripted agent, output in a temp dir
#   scripts/demo/record_demo.sh live   # paid: claude-sonnet-4-6, output in docs/media
#
# live reads ANTHROPIC_API_KEY, else VERITY_ANTHROPIC_API_KEY from .env. One recording
# asks three questions (~$0.05 at the mean cost of the answers of record); the API's
# spending budget keeps a session below $0.50 (scripts/demo/media.py). Model output
# varies between live takes: re-recording gives the same stack and questions, not the
# same frames. Needs Docker, Node 22 with corepack, uv and ffmpeg. Ports: 3000 (web,
# the API's CORS origin) and VERITY_API_PORT (default 8000).
set -euo pipefail

mode="${1:-}"
case "$mode" in
  live) out="${2:-docs/media}" ;;
  stub) out="${2:-$(mktemp -d)/verity-demo-stub}" ;;
  *) echo "usage: $0 live|stub [out-dir]" >&2; exit 2 ;;
esac

root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
for tool in docker corepack uv ffmpeg; do
  command -v "$tool" >/dev/null || { echo "missing: $tool" >&2; exit 2; }
done

# The key goes to compose alone, never to pnpm, next or the browser: it is held in a
# shell variable and the exported ones are removed. A stub run gets no key at all.
key=""
if [ "$mode" = live ]; then
  key="${ANTHROPIC_API_KEY:-}"
  if [ -z "$key" ] && [ -f .env ]; then
    key="$(sed -n 's/^VERITY_ANTHROPIC_API_KEY=//p' .env | tail -n 1 | tr -d "\r\"'")"
  fi
  [ -n "$key" ] || { echo "live needs ANTHROPIC_API_KEY" >&2; exit 2; }
fi
unset ANTHROPIC_API_KEY OPENAI_API_KEY

project=verity-demo
api="http://localhost:${VERITY_API_PORT:-8000}"
web=http://localhost:3000
compose=(docker compose -p "$project" -f docker-compose.yml -f scripts/demo/compose.demo.yml)
export VERITY_API_AUTH_DISABLED=true
# Off the default 5432 so a development database keeps running alongside.
export VERITY_DB_PORT="${VERITY_DB_PORT:-55433}"
playwright=(uv run --no-project --with playwright==1.64.1 python)
web_pid=""

cleanup() {
  [ -n "$web_pid" ] && kill "$web_pid" 2>/dev/null || true
  "${compose[@]}" down -v >/dev/null 2>&1 || true
}
trap cleanup EXIT

# A fresh volume each time: the corpus is exactly datasets/corpus, nothing uploaded.
"${compose[@]}" down -v >/dev/null 2>&1 || true
if curl -s -o /dev/null "$web"; then
  echo "port 3000 is already serving; stop it first" >&2
  exit 1
fi
ANTHROPIC_API_KEY="$key" OPENAI_API_KEY="" "${compose[@]}" up --build -d --wait

(cd web && corepack pnpm install --frozen-lockfile >/dev/null \
  && NEXT_PUBLIC_API_BASE="$api" corepack pnpm build >/dev/null)
# next itself, not pnpm, so that killing web_pid stops the server.
(cd web && exec node_modules/.bin/next start -p 3000 >/dev/null 2>&1) &
web_pid=$!
for _ in $(seq 60); do
  kill -0 "$web_pid" 2>/dev/null || { echo "next start exited" >&2; exit 1; }
  curl -fsS -o /dev/null "$web" && break
  sleep 1
done
curl -fsS -o /dev/null "$web" || { echo "web UI did not start on $web" >&2; exit 1; }

"${playwright[@]}" -m playwright install chromium >/dev/null
"${playwright[@]}" scripts/demo/record.py --out "$out" --web "$web" --api "$api" --expect "$mode"
ls -lh "$out"
