# Verity — dev shortcuts. Every target runs offline with no keys.
.PHONY: demo demo-web demo-api ui-install ui-test test all

# One-command demo path (S6):
#   1. FastAPI (stub agent, in-memory store, corpus preloaded) on :8000
#   2. Next.js dev server on :3000
# Ctrl-C stops both cleanly (trap on the API child).
demo: ui-install
	@echo "==> Starting Verity demo (API :8000 · web :3000, both stub-mode)"
	@bash -c 'trap "kill %1 2>/dev/null" EXIT; \
		uv run verity serve --host 127.0.0.1 --port 8000 & \
		cd web && NEXT_PUBLIC_API_BASE=http://localhost:8000 corepack pnpm dev'

# API only — for the demo transcript that shows API responses directly.
demo-api:
	uv run verity serve --host 127.0.0.1 --port 8000

# Web dev server only (API expected to be running elsewhere).
demo-web: ui-install
	cd web && NEXT_PUBLIC_API_BASE=http://localhost:8000 corepack pnpm dev

ui-install:
	@cd web && (test -d node_modules || corepack pnpm install --frozen-lockfile)

ui-test:
	cd web && corepack pnpm lint && corepack pnpm typecheck && corepack pnpm test && corepack pnpm build

test:
	uv run pytest -m "not integration"

all: test ui-test
