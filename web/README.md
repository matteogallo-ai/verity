# web/ — Next.js UI

The polished demo-ready interface (upload → ask → answer with highlighted source
citations + a confidence/chunks panel) is scoped to **v1.1** (S6). Scaffold B keeps the
differentiator — agentic pipeline, refusal, eval harness, observability — as the v1.0
deliverable; the UI is deliberately deferred so it lands when it serves the filmed demo
rather than as half-built chrome.

Contract it will consume: the API returns a full `Answer` (text, inline `citations` with
document char-offsets, `confidence`, `hits_used`, `usage`) — everything the UI needs to
highlight passages and render the confidence panel is already in `verity.types`.
