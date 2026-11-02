# 90-second demo script

Shows what runs offline with `make demo`: the API and the UI in **stub mode** (a
scripted agent, no API key, no LLM call). Every caption below matches what that mode
displays. The live-model numbers come from the committed scorecard, not from the
demo session.

## Setup

```bash
make demo          # FastAPI (stub agent, in-memory store) on :8000 + Next.js on :3000
```

The UI shows the stub-mode banner and a yellow `stub-agent` provenance badge on every
answer.

## Beats

### 0:00–0:10 — the claim
Open the README results table. Caption: *"Refuses when the evidence is weak.
Numbers from one live run (n = 16), committed as an artefact."*

### 0:10–0:35 — cited answer
Click the demo chip *"What is the notice period for terminating the master services
agreement?"*. The answer renders with an inline `[1]` citation; clicking it highlights
the clause in the source panel. The confidence readout shows `0.88`, labelled
"agent-reported (stub-agent)": the stub's canned value for on-topic questions.

Optional: upload one of the `datasets/corpus/*.md` files to show the upload path and
the chunk count.

### 0:35–0:55 — refusal
Click *"What is the CEO's home address?"*. The answer card becomes the refusal panel:
"I don't know based on the provided documents", confidence `0.15`, below the 0.55
threshold, with the stub's rationale and the hits it refused against.

### 0:55–1:25 — the evaluation artefacts
Terminal:

```bash
uv run verity eval run --ci                  # pipeline smoke test on stubs, ~10 s
uv run verity eval compare                   # diff across runs with matching provenance
jq '.refusal_calibration, .provenance' scorecards/live-v1.0.0.json
```

Show `per_refusal` (4 correct refusals, 1 out-of-scope question answered: q-015, see
the README's open issues) and `per_audit` (answer text, citations and per-claim judge
verdicts for each question). Caption: *"The stub run checks the plumbing. The README
numbers come from the live artefact."*

### 1:25–1:30 — close
Architecture diagram in the README, then the repo link.
