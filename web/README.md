# Verity UI

Next.js 15 (App Router) + TypeScript strict. The web surface for the Verity
demo: upload a document, ask a question, get a cited answer (or a calibrated
refusal), see the exact retrieved chunks and their rerank scores, and — the
point of the whole project — know at a glance **which agent produced the
answer** you're looking at.

## Zero-key demo path (~90 seconds)

The UI runs against the local FastAPI backend in **stub mode** by default —
no API keys required, no network egress, deterministic answers on the four
demo themes shipped with the corpus.

```bash
# One shell (starts API + web dev server together, stops both on Ctrl-C):
make demo
# → API on http://localhost:8000, UI on http://localhost:3000
```

Or manually, two shells:

```bash
# Shell 1 — API (stub agent, in-memory store, corpus preloaded):
uv run verity serve --host 127.0.0.1 --port 8000

# Shell 2 — Next dev server:
cd web
corepack enable
corepack pnpm install --frozen-lockfile
NEXT_PUBLIC_API_BASE=http://localhost:8000 corepack pnpm dev
```

Then walk the demo:

1. **Stub-mode banner** at the top: reads "Stub mode — no LLM key configured".
   This is the surface honesty guarantee — the UI never presents stub output as
   if it were a real model.
2. **Ask a demo question** from the chips: e.g. "What is the notice period for
   terminating the master services agreement?" — the answer text renders with
   an inline `[1]` citation marker. Clicking `[1]` highlights the source chunk
   in the right-hand panel (and vice-versa).
3. **Ask an out-of-scope question**: e.g. "What is the CEO's home address?" —
   the answer card becomes a first-class refusal panel with the rationale and
   the hits the agent chose to refuse against.
4. **Upload a document** (`.md` / `.txt` / `.pdf` / `.docx`) — it becomes part
   of the queryable corpus immediately. The document id is content-addressed:
   the same bytes always produce the same id, regardless of filename.

## Provenance is always visible

Every answer renders a **provenance badge** sourced verbatim from
`AskResponse.provenance.agent_model`. The API derives that value per-request
from the synthesizer's `Completion` (`Answer.provider_used` / `model_used`),
so a live-mode routing fallback (Anthropic → OpenAI on
`ProviderUnavailableError`) is reflected honestly. The client never infers
provenance and never defaults from Settings.

- `stub-agent` (yellow chip) — the scripted, deterministic stub. Demo default.
- `claude-sonnet-4-6`, `gpt-4.1-mini`, … (green chip) — the model the router
  actually served THIS request with.

If the API is unreachable, the UI stops rendering answers — never a cached
badge, never a plausible-looking placeholder.

## What the UI never invents

Rule of iron: every number on screen has a direct field in the API response.

- `latency_ms` → `answer.usage.latency_ms` (wall-clock, real).
- rerank score per chunk → `answer.hits_used[i].score` (float from the
  cross-encoder, verbatim, 3-decimal precision).
- `cited` flag per chunk → derived structurally from
  `answer.citations` set-membership on the API side.
- `confidence.score` → **passthrough** of `answer.confidence.score`. Rendered
  by `ConfidenceReadout` with an explicit "agent-reported (stub-agent)" or
  "agent-reported (`<model>`)" label AND the underlying signals a careful
  reader would want (`refused`, `citations` count, top rerank score, evidence
  hits). Never a bare authoritative percentage next to an answer — always
  provenance-labelled with signals visible.

No composite "confidence percentage" is invented, ever. Values that aren't
returned by the API render as `pending` or `n/a` — not zeroed, not guessed.

## Verbatim citations (invariant, enforced by test)

Every `citations[i].quote` returned by the API is a **verbatim substring** of
the source document at `[char_start:char_end]`. This is stacked-enforced:
(1) the synthesizer drops any citation whose quote isn't a substring of the
referenced chunk (`chunk.text.find(quote) >= 0`);
(2) offsets are re-anchored to `document.text` via the S1 chunker's own
invariant. The stub agent by construction only cites literal anchor phrases
present in the retrieved evidence. `tests/unit/test_citation_verbatim_invariant.py`
walks the whole answerable set and asserts `document.text[cs:ce] == quote` on
every emitted citation.

## Live mode (S7)

Setting `VERITY_ANTHROPIC_API_KEY` (or `VERITY_OPENAI_API_KEY`) and
restarting `verity serve` switches the API into live mode. The UI picks it up
automatically on the next `/status` poll: the banner disappears, the badge
turns green and reads the actual model id.

The published live scorecard — real faithfulness / hallucination / refusal
numbers on a real LLM agent — lands at S7.

## Development

```bash
corepack enable                                # once — pins pnpm@9.15.0 via packageManager
corepack pnpm install --frozen-lockfile
corepack pnpm dev            # dev server on :3000
corepack pnpm test           # vitest — deterministic, no network
corepack pnpm typecheck      # tsc --noEmit strict
corepack pnpm lint           # next lint (eslint 9 flat config)
corepack pnpm build          # next build (production bundle)
```

The `web` job in `.github/workflows/ci.yml` runs all of the above with a
frozen lockfile on Node 22.

## Layout

```
web/
├── app/
│   ├── globals.css
│   ├── layout.tsx
│   └── page.tsx                 # the two-panel app shell
├── components/
│   ├── AnswerCard.tsx           # cited answer + inline citation markers
│   ├── AskBox.tsx               # question box + demo chips
│   ├── ProvenanceBadge.tsx      # stub-agent / live-<model>
│   ├── RefusalState.tsx         # first-class refusal panel
│   ├── SourcePanel.tsx          # honest chunk list: score + cited flag
│   ├── StubModeBanner.tsx       # persistent iff provenance.is_stub
│   └── UploadPanel.tsx
├── lib/
│   ├── api.ts                   # typed fetch client (fetchStatus/ask/ingest)
│   ├── highlight.ts             # segmentAnswer(text, citations) → segments
│   └── types.ts                 # 1:1 mirror of verity/api/models.py
├── tests/
│   ├── AnswerCard.test.tsx      # cited + refusal renders, provenance badge, click marker
│   ├── HighlightInteraction.test.tsx  # citation ↔ chunk bidirectional highlight
│   ├── SourcePanel.test.tsx     # cited/not-cited, score display, empty state
│   ├── StubModeBanner.test.tsx  # iff provenance.is_stub
│   ├── fixtures.ts              # real API-shape samples
│   ├── highlight.test.ts        # segment algorithm
│   └── setup.ts
├── eslint.config.mjs            # ESLint 9 flat config
├── next.config.mjs
├── package.json                 # pnpm@9.15.0 pinned via packageManager
├── tsconfig.json                # strict, path alias @/*
└── vitest.config.ts             # jsdom, coverage v8
```

## Non-goals in S6

- No Playwright/E2E — flake budget stays low for now. Candidate for S7 once
  the live-agent demo path exists.
- No auth, no multi-tenant, no deployment config beyond local dev.
- No fabricated confidence numbers, no invented composite scores. Ever.
