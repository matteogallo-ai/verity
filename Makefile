# Verity — dev shortcuts. Every target runs offline with no keys.
.PHONY: demo demo-cct demo-cct-args demo-web demo-api db ui-install ui-test test test-fast test-slow mutation all

# Blank provider keys override a key in .env, which would switch the API to the live,
# paid agent.
BLANK_KEYS = VERITY_ANTHROPIC_API_KEY= VERITY_OPENAI_API_KEY=
STUB_ENV = VERITY_API_AUTH_DISABLED=true $(BLANK_KEYS)

# One-command demo path:
#   0. Postgres + pgvector (docker compose service db), migrations applied
#   1. FastAPI (stub agent, pgvector store, corpus ingested on first request) on :8000
#   2. Next.js dev server on :3000
# Ctrl-C stops both cleanly (trap on the API child).
demo: ui-install db
	@echo "==> Starting Verity demo (API :8000 · web :3000, both stub-mode)"
	@bash -c 'trap "kill %1 2>/dev/null" EXIT; \
		$(STUB_ENV) uv run verity serve --host 127.0.0.1 --port 8000 & \
		cd web && NEXT_PUBLIC_API_BASE=http://localhost:8000 corepack pnpm dev'

# Demo on one collective agreement, in French, for a client meeting:
#   make demo-cct CORPUS=<folder of the agreement> [LIVE=1] [CORPUS_NAME="CCT Métal-Vaud"] [UI_DEV=1]
# The API ingests CORPUS in memory under profiles/demo-cct-fr.env while the web UI
# is built, then served on :3000; the UI answers once /ready does. Only LIVE=1 keeps the provider key
# (from .env or the shell), and every answer is then paid; any other value, LIVE=0
# included, blanks the keys and the scripted stub answers.
LIVE_DEMO = $(filter 1,$(LIVE))
# The UI is a production build (next build, then next start): the dev server shows the
# Next.js badge in front of the client. UI_DEV=1 keeps the dev server, with hot reload.
# NEXT_PUBLIC_API_BASE goes to the build, where Next inlines it.
CCT_WEB = $(if $(filter 1,$(UI_DEV)),corepack pnpm dev,corepack pnpm build && corepack pnpm start)
# The title of the presentation view; without CORPUS_NAME it reads only "Assistant CCT".
# The name is passed as written: French titles hold apostrophes, and the recipe runs in
# a single-quoted bash -c, so each quote is escaped once for the inner assignment and
# once for the bash -c string. $(value) keeps a "$" from being expanded by make.
SQ := '
escape_sq = $(subst $(SQ),$(SQ)\$(SQ)$(SQ),$(1))
CORPUS_NAME_ENV = $(if $(value CORPUS_NAME),$(call escape_sq,VERITY_CORPUS_NAME=$(SQ)$(call escape_sq,$(value CORPUS_NAME))$(SQ)),)

demo-cct: demo-cct-args ui-install
	@echo "==> CCT demo on $(CORPUS) (API :8000 · web :3000, $(if $(LIVE_DEMO),live: answers are paid,stub))"
	@bash -c 'trap "kill %1 2>/dev/null" EXIT; \
		set -a; . profiles/demo-cct-fr.env; set +a; $(if $(LIVE_DEMO),,$(BLANK_KEYS)) $(CORPUS_NAME_ENV) uv run verity serve --host 127.0.0.1 --port 8000 --corpus "$(CORPUS)" & \
		cd web && NEXT_PUBLIC_API_BASE=http://localhost:8000 $(CCT_WEB)'

# Before ui-install, so a missing folder fails at once.
demo-cct-args:
	@test -n "$(CORPUS)" || { echo "usage: make demo-cct CORPUS=<folder> [LIVE=1] [CORPUS_NAME=<title>] [UI_DEV=1]"; exit 2; }
	@test -d "$(CORPUS)" || { echo "$(CORPUS): no such folder"; exit 2; }

# API only — for the demo transcript that shows API responses directly.
demo-api: db
	$(STUB_ENV) uv run verity serve --host 127.0.0.1 --port 8000

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
