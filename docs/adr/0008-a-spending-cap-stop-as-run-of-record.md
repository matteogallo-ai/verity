# ADR 0008 — A run stopped by its spending cap can stand as the run of record

**Status:** proposed · 2026-11. The owner accepts it by merging the v1.1.0 release PR.
Until then, the rule of `ScoreCard.incomplete_reason` stands as written before it.

## Context

`ScoreCard.incomplete_reason` said an incomplete run "never stands as a run of
record". This covered two cases, which differ:

- **Lost questions.** A run can lose questions to errors (provider or store down). Its
  metrics then skip questions the system failed on.
- **Questions never asked.** A run can stop at its spending cap. The questions after
  the stop were never asked; every question that was asked has its record.

The v1.1.0 test run, `eval_513b839_test.incomplete-20261107T122042441670Z.json`, is of
the second kind:

- It stopped at its $2.40 cap after 54 of 56 questions, with 0 errored.
- `q-094` and `q-095`, both unanswerable and tagged `adversarial`, were not run.
- The cap was set from a judge cost per question asked, not per judged answer: see
  "Deviations after the runs" in `docs/preregistration/two-arm-comparison.md`.

## Decision

A run can stand as the run of record when all of these hold:

- it stopped at its spending cap (`budget_exceeded`) with no errored question;
- every rendering of it states "n of N", names the questions not run, and counts them
  neither as errors nor as queries (`sample_sizes.n_not_run`);
- its metrics are computed over the questions it asked, and the limits section says
  which questions are missing.

A run that lost questions to errors still never stands as a run of record.

## Alternatives

- **Rerun the test split with a larger cap.** Estimate: 56 questions × $0.0455 per
  query with the judge = $2.548; cap $2.90.
  - Not run: it would be a second test run, chosen after seeing the first.
  - It would also bring this mission's spend from $8.8931 to at most $11.79 plus one
    question's overshoot, against a $12 ceiling.
  - Either reason makes it the owner's decision.
  - If the owner chooses it, this ADR is withdrawn and the new run replaces the
    scorecard.
- **Keep `969e1d5` as the run of record.** Rejected: it predates the shipped defaults,
  the datamarked synthesis prompt and judge prompt v0.2.0, so it does not measure the
  release.

## Consequences

- The v1.1.0 figures rest on 54 questions, 36 answerable and 18 unanswerable.
- Two unanswerable questions are missing. A complete run would have refusal recall
  between 13/20, if neither were refused, and 15/20, if both were.
