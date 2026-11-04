# Figure rule for the support check: measurement, 2026-11-03

## Question

The second neutral audit (M3) found that the support check's word coverage passes an
answer with a wrong number, date or amount. The proposed rule: an answered draft is
refused when the answer contains a figure (a number, or a month next to a day number)
that none of its cited quotes contains. The condition set for activating it: on the
existing live artefacts, it refuses no correct answer.

**Result: the condition is not met, so the rule is not active.** Both variants below
refuse correct answers. The support check is unchanged.

## Method

- Script: [`scripts/measure_figure_rule.py`](../../scripts/measure_figure_rule.py),
  tested in `tests/unit/test_measure_figure_rule.py`. No model call.
- Data: `answer_text`, `question` and `citations[].quote` of the answered
  (non-refused) examples in `per_audit` of five live artefacts. Only the first is on
  `main`; the others are on `fix/audit-part-c`, as for the F-8 measurement.
- Variants:
  - **strict**: every figure of the answer must be in a cited quote;
  - **exempt**: a figure is also accepted when it is in the question, when it is
    preceded by a label (`No.`, `Section`, `Amendment` …), or when one arithmetic step
    (+, −, ×, ÷, a percentage) or the sum of any three numbers from the quotes and the
    question gives it, within 0.5 %. Any number glued to a letter or a hyphen is
    skipped as an identifier: `FY2025` and `NW-2025-001`, but also `60-day`,
    `12-month`, `24h` and `$40k`. This variant is therefore much more lenient than "an
    identifier is exempt": `90 days` passes against quotes holding `30` and `60`.
- What is compared is the set of numbers, not whole figures: a date counts as its
  month name plus a separate day number, `2:00` and `02:00` differ, and a comma
  decimal such as `2,5` is not read as a number.
- Correctness of each refused answer was read against the dataset's reference answer.
  Every one listed below was rated faithful (1.0) by the judge.

## Results

| Artefact | Answered | Strict would refuse | Exempt would refuse |
|---|---:|---|---|
| `live/eval_d0ca1d5_questions.json` | 12 | q-013 | q-013 |
| `live/eval_6ec544d_dev.json` | 21 | q-001, q-002, q-054, q-057, q-060, q-065, q-087 | q-002, q-060 |
| `noise/eval_6ec544d_dev.rerun2.json` | 20 | q-001, q-002, q-026, q-054, q-057, q-060, q-065, q-087 | q-026, q-057, q-060 |
| `noise/eval_6ec544d_dev.rerun3.json` | 22 | q-001, q-002, q-026, q-054, q-057, q-060, q-065, q-087 | q-026, q-060 |
| `live/eval_969e1d5_test.json` | 34 | q-010, q-029, q-037, q-053, q-056, q-062, q-066 | q-010, q-037 |

Every listed answer is correct except q-087, which is out of scope and should have been
refused. The strict variant catches it, and the exempt variant lets it through. A rule
lenient enough to accept the correct answers above also accepts wrong figures it was
meant to catch; what the exempt variant still catches was not measured.

Why correct answers fail the strict variant:

- **Computed figures.** The answer states a result the quotes imply, as the reference
  answers do: q-029 "40% of the fee … US$34,000", q-037 "$40 + $75 + $40 = $155",
  q-056, q-065, q-054, q-066.
- **Figures from the question.** q-062 "A US$75,000 purchase …".
- **Identifiers.** q-001 and q-002 "Amendment No. 1", q-053 "FY2024", q-057 "FY2025".

Why correct answers still fail the exempt variant:

- **A number written in words in the quote.** q-010: "24 hours" against the quote's
  "twentyfour hours".
- **A sum that uses one quoted number twice.** q-037: $40 is quoted once and counted on
  two days.
- **A correct figure the citations do not carry.**
  - q-002: the original $1,000,000 cap, from a chunk that was not cited.
  - q-013: the 02:00–06:00 window.
  - q-026: the NDA date.
  - q-057 and q-060: the board meeting date and the quarter end.

  These answers are right but say more than their quotes. Refusing them is a policy
  choice ("an answer states only what it cites"), not a correctness fix.

## Decision for the owner

This leaves three options:
1. Keep the support check as it is (the current state).
2. Adopt the policy "every figure is quoted" and accept the refusals above.
3. Change the synthesis prompt so that figures are always quoted, which needs a paid
   run to measure.
