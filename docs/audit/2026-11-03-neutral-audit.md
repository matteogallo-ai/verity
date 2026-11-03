# Neutral audit of `main`, 2026-11-03

## Method

- **Target**: `main` at `f6c8462`, in a clean worktree. The Part C branch
  (`fix/audit-part-c`) was out of scope.
- **Auditor**: a fresh-context subagent with no knowledge of the project's history. Its
  brief was to act as a staff engineer deciding whether to hire the author; to judge the
  code, tests and docs on what they do, not on what they claim; to rank findings by
  severity with verified evidence; to make no paid or network call; and to treat
  repository content as data.
- **Raw output**: [`2026-11-03-neutral-audit-findings.md`](2026-11-03-neutral-audit-findings.md),
  the auditor's report as written.

**Verdict: lean hire.** The auditor recomputed the published numbers from the committed
audit trail and found that they match it. It found the limits stated plainly. Its
concerns were defects in the least visible parts: the eval judge, routing, async I/O and
the support check. Several of them break the repository's own rules.

The checks the auditor ran on `f6c8462` all passed:
- ruff: clean;
- strict mypy: clean;
- fast lane: 646 passed;
- `check_claims.py --no-run`: 0 problems;
- `generate_headline_results.py --check`: up to date.

## What became of each finding

Each finding was reproduced before acting on it. Where a fix changes behaviour, the test
was written first and seen failing.

| # | Sev. | Finding | Outcome | Evidence |
|---|---|---|---|---|
| F-1 | HIGH | The judge prompt has no delimiters and no untrusted-content instruction; the test named "every prompt marks its inputs untrusted" skips it | **Confirmed. Prompt unchanged by decision, gap made visible** | This gap is recorded in `docs/THREAT_MODEL.md` (LLM01 on the evaluator). A defended prompt is a new judge version, with its own rejudge and κ, so `prompts/judge_faithfulness.prompt` stays v0.1.0 here. A strict expected failure, `test_the_judge_prompt_marks_its_inputs_untrusted`, turns red when the prompt is defended. C24 is narrowed to the answering path (9d5fd2b). |
| F-2 | MED | No fallback when the first provider has no key | **Fixed** | Keyless providers are left out of the route, and construction fails when no provider has a key. Two tests were red before the change (a8a4def). |
| F-3 | MED | The headline predates the current pipeline; the gate's thresholds were fitted on the same 16 questions | **Fixed in the docs; measurement open** | The README open issues and the system card now say so (22364d1). A held-out measurement needs a paid run. Part C has dev/test runs of the current pipeline. |
| F-4 | MED | The judge parser counts `"false"` as supported | **Fixed** | Verdicts must be booleans or `true`/`false`; anything else is malformed. Seven cases were red before the change (1e09bc2). |
| F-5 | MED | The stored scorecard keeps the diluted mean; two copies of the scoped computation exist | **Fixed in Part C, not on `main`** | `77c9c75` on `fix/audit-part-c` adds `verity.eval.answer_quality` as the one definition. It depends on earlier Part C commits (`load_run`), so it was not ported alone: the cherry-pick failed to collect `test_live_artefacts.py`. |
| F-6 | MED | A paid eval run loses everything on a judge error or a budget abort; errored spend is dropped | **Partly fixed** | A judge outage now errors its example instead of ending the run (dc9528f, test red first). The errored answers' spend is fixed in Part C (`337455c`). Still open: a budget abort persists nothing. Writing a partial artefact changes the run-store contract (a partial run must not read as a run of record), which is a design decision for the owner. |
| F-7 | MED | PDF/DOCX conversion blocks the event loop | **Fixed** | Model loading and conversion run in `asyncio.to_thread`. A test where a ticker gets no turn during a 0.5 s conversion was red (0 ticks) before the change (1bd174c). |
| F-8 | MED | The support check ignores negation; its injection markers miss paraphrases and flag benign text | **Confirmed. Not fixed by a lexical rule; documented and pinned** | A polarity rule ("an answer's negation must appear in its quotes") was measured on the five live artefacts. It refused correct answers: test q-064 "No, … explicitly prohibited", and q-001, where "Amendment No. 1" reads as a negation. It also refused two out-of-scope answers (q-015, q-087). Meaning therefore stays with the confidence scorer. The module docstring states both limits, and tests pin them (a1eece9). |
| F-9 | LOW | The database URL is a plain `str` | **Fixed** | Cherry-picked from Part C `13f60ca`. The conflict in `db upgrade` was resolved without Part C's error redaction (4afb5ba). |
| F-10 | LOW | One connection per operation; a docstring claims a pool | **Docstrings fixed; pool not added** | The docstrings now say there is no pool, and that each `/ask` opens two connections per sub-question (364ab17). Connection setup has not been measured on pgvector: the logged live runs used the in-memory store, where retrieval is about 6 % of answer time. A pool adds a dependency and a lifecycle to the API, so it waits for a measurement. |
| F-11 | LOW | The methodology names per-1k prices in the wrong module | **Fixed** | 7315238 |
| F-12 | LOW | `< /document>` escapes delimiter neutralisation | **Fixed** | Two cases were red before the change (c7e4834). |
| F-13 | LOW | Leftover `_ = cast` lines with misleading comments | **Fixed** | ruff, mypy and vulture are clean without them (901301f). |

## Not settled by this audit

- **F-1.** The defended judge prompt v0.2.0 and its adversarial test exist on a local
  branch. Rejudging the same 60 claims and computing the new κ waits for the owner's
  budget decision.
- **F-3.** No published number describes the current pipeline on held-out questions.
- **F-6.** What a budget abort writes to disk is not decided.
- **F-10.** No connection pool.
