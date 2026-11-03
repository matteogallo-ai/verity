# Verity — dev shortcuts. Every target runs offline with no keys.
.PHONY: demo demo-web demo-api db ui-install ui-test test test-fast test-slow mutation all

# One-command demo path:
#   0. Postgres + pgvector (docker compose service db), migrations applied
#   1. FastAPI (stub agent, pgvector store, corpus ingested on first request) on :8000
#   2. Next.js dev server on :3000
# Ctrl-C stops both cleanly (trap on the API child).
demo: ui-install db
	@echo "==> Starting Verity demo (API :8000 · web :3000, both stub-mode)"
	@bash -c 'trap "kill %1 2>/dev/null" EXIT; \
		VERITY_API_AUTH_DISABLED=true uv run verity serve --host 127.0.0.1 --port 8000 & \
		cd web && NEXT_PUBLIC_API_BASE=http://localhost:8000 corepack pnpm dev'

# API only — for the demo transcript that shows API responses directly.
demo-api: db
	VERITY_API_AUTH_DISABLED=true uv run verity serve --host 127.0.0.1 --port 8000

# Postgres + pgvector, the demo's store. --wait returns once the healthcheck passes,
# so the migrations never race the server start.
db:
	docker compose up -d --wait db
	uv run verity db upgrade

# Web dev server only (API expected to be running elsewhere).
demo-web: ui-install
	cd web && NEXT_PUBLIC_API_BASE=http://localhost:8000 corepack pnpm dev

ui-install:
	@cd web && (test -d node_modules || corepack pnpm install --frozen-lockfile)

ui-test:
	cd web && corepack pnpm lint && corepack pnpm typecheck && corepack pnpm test && corepack pnpm build

test: test-fast test-slow

test-fast:
	HF_HUB_OFFLINE=1 uv run pytest -m "not integration and not slow"

test-slow:
	uv run pytest -m "slow and not integration"

# Mutation testing of the modules listed under [tool.mutmut] in pyproject.toml, against
# the fast lane. Starts from a clean copy so the score is not a stale cache; the
# measured result is recorded in docs/quality/mutation.md.
mutation:
	rm -rf mutants
	OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES HF_HUB_OFFLINE=1 uv run mutmut run
	uv run mutmut results --all true

all: test ui-test
