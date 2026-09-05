# Contributing to Verity

Thanks for looking. Verity optimises for one thing: a senior engineer opening the repo
sees production maturity — measured quality, refusal over hallucination, tests, CI,
observability. Contributions are held to that bar.

## Ground rules

1. **Every change keeps CI green.** Ruff (lint + format), mypy `--strict`, and the test
   suite must pass. No exceptions, no skips-to-ship.
2. **New behaviour ships with tests.** Unit for logic, integration (marked
   `@pytest.mark.integration`) for anything touching Postgres or the network.
3. **Types are not optional.** Full type hints; new public surfaces speak in
   `verity.types`. `Any` needs a comment justifying it.
4. **No fabricated metrics.** Numbers in the README come from a real eval run tied to a
   git SHA. If you change the pipeline, re-run `verity eval run` and update the scorecard.
5. **Refusal is a feature, not a bug.** Changes that raise coverage by making the system
   answer when it shouldn't will be rejected. Watch refusal precision/recall.

## Workflow

```bash
uv sync --dev
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest -m "not integration"
```

Commits follow Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`,
`chore:`). Open a PR against `main`; describe the change and, if it touches the pipeline,
paste the before/after scorecard.
