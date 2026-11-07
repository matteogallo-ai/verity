# Results

The detail behind the README's key result: the full results table of the run of
record, the judge's validation against blind human labels, and what the numbers do not say.

## Measured results

<!-- RESULTS:START -->
**Measured on**: `agent=claude-sonnet-4-6 (56/56)` · `judge=claude-opus-5-5 (34/56 answered — 22 refusals bypass the judge)` · `embedding=BAAI/bge-small-en-v1.5` · commit `969e1d5` · 2026-11-02 · dataset `test` · n=56 (36 answerable + 20 unanswerable) · **0 errored** (kept out of every metric) · total cost **$1.2768**

| Metric | Value | Notes |
|---|---:|---|
| **Invented answers** | **1/34** | ↓ better · answers with a claim the judge finds unsupported / every answer shipped, answerable or not: 1/34 · 95 % CI [0.005, 0.149] · `q-074` · judge κ 0.71 [0.49, 0.89] vs blind human, validated on 60 claims of which 1 real hallucination |
| **Refusal recall** | **0.850** | unanswerable questions refused / unanswerable: 17/20 · 95 % CI [0.640, 0.948] · threshold 0 |
| **Refusal precision** | **0.773** | refusals that were unanswerable / refusals: 17/22 · 95 % CI [0.566, 0.899] |
| Faithfulness, per question | 0.990 | mean over 34 judged answers · judge κ 0.71 [0.49, 0.89] vs blind human, validated on 60 claims of which 1 real hallucination |
| Faithfulness, per claim | 0.991 | pooled over claims: 108/109 · 95 % CI [0.950, 0.998] · judge κ 0.71 [0.49, 0.89] vs blind human, validated on 60 claims of which 1 real hallucination |
| Citation accuracy, per question | 0.968 | mean over 34 judged answers · judge κ 0.71 [0.49, 0.89] vs blind human, validated on 60 claims of which 1 real hallucination |
| Citation accuracy, per claim | 0.963 | pooled over claims: 105/109 · 95 % CI [0.909, 0.986] · judge κ 0.71 [0.49, 0.89] vs blind human, validated on 60 claims of which 1 real hallucination |
| Hallucination rate, answerable only | 0.000 | ↓ better · answers with an unsupported claim / answered answerable questions (an answer to an unanswerable question is outside it): 0/31 · 95 % CI [0.000, 0.110] |
| Unanswerable answered | 3/20 | ↓ better · an answer shipped where the corpus has none: 3/20 · 95 % CI [0.052, 0.360] |
| Retrieval nDCG@8 | 0.929 | scored without an LLM on hits retrieved for the LLM sub-questions |
| Retrieval recall@8 | 0.986 | as nDCG |
| Retrieval precision@8 | 0.146 | k>gold ceiling; interpret with recall |
| Latency p50 / p95 / p99 | 5445.7 ms / 8949.3 ms / 10929.3 ms | warm; cold start `8460.1 ms` measured separately |
| Cost / query | $0.0132 | $0.0132 agent only · $0.0228 with the judge · total $1.2768 = agent $0.7408 + judge $0.5360 |

Refusals by cause: `irrelevant_citation` 4 (4 answerable) · `low_citation_coverage` 7 (1 answerable) · `model_refused` 2 (0 answerable) · `no_citation` 6 (0 answerable) · `suspected_injection` 3 (0 answerable).
No refusal has the cause `below_threshold`: the confidence threshold decided none of them.

Generated from [`scorecards/live-v1.1.0.json`](../scorecards/live-v1.1.0.json), itself computed from the run artefact `datasets/eval/runs/live/eval_969e1d5_test.json` (`scripts/generate_headline_results.py --check` recomputes both).

<!-- RESULTS:END -->

The v1.0.0 artefact
[`eval_d0ca1d5_questions.json`](../datasets/eval/runs/live/eval_d0ca1d5_questions.json)
is immutable and still stores citation accuracy **0.7917**, under the earlier
definition that averaged its 4 refusals in at 1.0. Under the single definition used
above (refusals excluded) it is **0.7222**, per question over 12 judged answers; the
recomputation is in
[`eval_d0ca1d5_questions.recomputed.json`](../datasets/eval/runs/live/eval_d0ca1d5_questions.recomputed.json)
(`scripts/recompute_live_aggregates.py`).

Every value in the table is copied from the committed scorecard JSON by
[`scripts/generate_headline_results.py`](../scripts/generate_headline_results.py);
`--check` fails CI if this table or the README's key result drifts. Each answer-quality metric carries its
`scope` and `n_denominator`. The full audit trail (per-question answer, citations,
refusal cause, confidence score, per-claim judge verdict) is
[`datasets/eval/runs/live/eval_969e1d5_test.json`](../datasets/eval/runs/live/eval_969e1d5_test.json).
The run used the in-memory store (BM25 sparse arm); `verity eval run` defaults to
pgvector (`--store memory` reproduces it). Definitions:
[`docs/evaluation-methodology.md`](evaluation-methodology.md).
Each proportion with its k/n and Wilson 95 % interval, the approved SLOs and whether
this run meets them, the intended use and the known limits:
[`docs/SYSTEM_CARD.md`](SYSTEM_CARD.md), generated from the same scorecard.

## Judge validation

The answer-quality rows above are judged by a model. Its agreement with blind human
labels is the result of record; an adjudication of the disagreements, published
separately, does not replace it.

<!-- JUDGE:START -->
Judge `claude-opus-5-5` with judge prompt v0.1.0 (`datasets/eval/judge_validation/judge_faithfulness.v0.1.0.prompt`, sha256 `2c5f8eca49ba`) against **blind human labels**, the reference: 60 claims labelled without origin or judge verdict, 40 real claims (16 from the dev run, 24 from the test run) and 20 claims perturbed to be false (one number, date, negation or entity changed). The κ below belongs to this prompt version: a changed judge prompt is a new judge and needs its own κ.

The judge is validated on 60 claims, of which 1 real hallucination (a real claim the human finds unsupported): its sensitivity to real hallucinations rests on that count, the perturbed claims bound it from above.

| Measure | Value | n · notes |
|---|---|---|
| Cohen's κ, all claims | 0.71 · bootstrap 95 % CI [0.49, 0.89] | 60 claims · observed agreement 0.883 (53/60 · 95 % CI [0.778, 0.942]) · 46 supported / 14 unsupported by the human |
| Cohen's κ, real claims only | 0.48 · bootstrap 95 % CI [0.00, 1.00], κ undefined in 206 of 5000 resamples | 40 claims · observed agreement 0.950 (38/40 · 95 % CI [0.835, 0.986]) · **fragile**: only 1 of 40 claims is in the smaller human class, so one or two disagreements move κ widely |
| Judge specificity on real claims (human: supported) | 0.949 | 37/39 · 95 % CI [0.831, 0.986] |
| Sensitivity: perturbed claims the judge finds unsupported (**upper bound**) | 0.800 | 16/20 · 95 % CI [0.584, 0.919] · by kind: date 5/5 · entity 3/5 · negation 5/5 · number 3/5 |
| … among perturbed claims the human finds unsupported | 0.923 | 12/13 · 95 % CI [0.667, 0.986] |
| Perturbed claims the human finds unsupported | 0.650 | 13/20 · 95 % CI [0.433, 0.819] |

Disagreements: the judge rejects 6 claims the human accepts and accepts 1 the human rejects. Notes from the artefact:

- kappa has no Wilson interval: it is not a proportion; with 60 items it is uncertain by itself.
- a perturbed claim counts as detected if any sub-claim the judge made of it is unsupported, even outside the changed span: sensitivity is an upper bound (per-claim detail in perturbed_verdicts.json).
- robustness to injected content is not measured: judge prompt v0.1.0 puts the retrieved text in front of the judge without delimiters or an untrusted-content instruction ([THREAT_MODEL](THREAT_MODEL.md#llm01-prompt-injection)), and only 4 of the 60 claims had the corpus note written to carry an injection in the judge's context.

Generated from [`datasets/eval/judge_validation/agreement.json`](../datasets/eval/judge_validation/agreement.json) and `kappa_ci.json` (`scripts/render_judge_validation.py --check`); the labels and the key are pinned in `datasets/eval/judge_validation/SHA256SUMS`.
<!-- JUDGE:END -->

### Adjudicated agreement (not the result of record)

<!-- JUDGE-ADJUDICATED:START -->
10 claims re-read with both verdicts visible, by the annotator of the blind labels: every disagreement, and every perturbed claim judged true blind. 7 labels changed, **6 of them to the judge's verdict**. The judge's verdict was in view, so a change towards it may be persuasion as much as correction: these figures bound agreement from above and validate nothing. Of the 7 perturbed claims judged true blind, 2 stay true after the change and 5 were annotator misses.

| Measure | Blind (of record) | Adjudicated |
|---|---|---|
| Cohen's κ, all claims | 0.71 | 0.93 |
| Cohen's κ, real claims only (fragile: 3 unsupported) | 0.48 | 1.00 |
| Observed agreement, all claims | 53/60 · 95 % CI [0.778, 0.942] | 58/60 · 95 % CI [0.886, 0.991] |
| Perturbed claims the human finds unsupported | 13/20 · 95 % CI [0.433, 0.819] | 18/20 · 95 % CI [0.699, 0.972] |
| … of which the judge finds unsupported | 12/13 · 95 % CI [0.667, 0.986] | 16/18 · 95 % CI [0.672, 0.969] |

Generated from [`datasets/eval/judge_validation/agreement_adjudicated.json`](../datasets/eval/judge_validation/agreement_adjudicated.json); each decision and its reason is in `adjudication.csv`.
<!-- JUDGE-ADJUDICATED:END -->

### Judge prompt v0.2.0 (not the judge of the runs of record)

<!-- JUDGE-V2:START -->
Judge `claude-opus-5-5` with judge prompt v0.2.0 (sha256 `aa66e62370a2`), the prompt in `prompts/`: its inputs are delimited and its evidence datamarked, and it names the citation of each claim. Rejudged against the same blind labels, for $1.3045, on **52 of 60 claims only: the budget stopped the rejudge before 8**, so the figures below are incomplete and both prompts are compared on those claims; the unjudged ones are the last in id order, not a random sample. Every claim, real or perturbed, is judged as a one-claim answer; v0.1.0 used the live whole-answer verdict for the real claims, so a change on real claims mixes the prompt and the protocol. The runs of record were judged by v0.1.0: their answer-quality figures carry the v0.1.0 κ above.

| Measure, on the 52 claims v0.2.0 judged | v0.1.0 (runs of record) | v0.2.0 |
|---|---|---|
| Cohen's κ, all claims | 0.64 | 0.60 |
| Observed agreement | 45/52 · 95 % CI [0.747, 0.933] | 44/52 · 95 % CI [0.725, 0.920] |
| Cohen's κ, real claims only (fragile: 1 unsupported) | 0.48 | 0.37 |
| Judge specificity on real claims | 33/35 · 95 % CI [0.814, 0.984] | 32/35 · 95 % CI [0.776, 0.970] |
| Sensitivity on perturbed claims (upper bound) | 12/16 · 95 % CI [0.505, 0.898] | 12/16 · 95 % CI [0.505, 0.898] |

On all 60 claims v0.1.0 has κ 0.71, the figure of record above.

The judge named a valid citation it found supporting for 40 of the 57 claims it returned. The blind labels judge support only, so this count is not validated against a human.

Generated from [`datasets/eval/judge_validation/judge-v0.2.0/agreement.json`](../datasets/eval/judge_validation/judge-v0.2.0/agreement.json); per-claim verdicts in `verdicts.json` beside it.
<!-- JUDGE-V2:END -->

## Open issues: what the numbers do not say

- **The runs predate the current prompts.** Both v1.1.0 runs used
  `synthesize.prompt` v0.2.0 and judge prompt v0.1.0: the datamarked synthesis (C33)
  and the delimited judge came after them, and no live run of the current pipeline is
  published here.
- **The threshold decides nothing on this data.** Every refusal above has a
  deterministic cause (no citation, low citation coverage, irrelevant citation,
  suspected injection) or is the scorer's own refusal; none is `below_threshold`.
  The threshold was chosen on the dev split by a rule fixed before the run (highest
  refusal recall at precision ≥ 0.9, else highest F1): no threshold reached 0.9,
  refusal precision and recall are identical for every threshold from 0.00 to 0.95,
  and the rule gave 0.00. Sweep and figure:
  [`scorecards/refusal-analysis.json`](../scorecards/refusal-analysis.json),
  [`docs/figures/reliability.png`](figures/reliability.png).
- **Refusal precision is a known limit** (row above, with its interval). Every
  refusal of an answerable question comes from the citation checks
  (`irrelevant_citation`, `low_citation_coverage`: counts under "Refusals by cause");
  dev shows the same cause (`scorecards/refusal-analysis.json`). Tuning those checks
  is done on dev only; a new test run belongs to a new version.
- **The confidence score is not calibrated.** On test, 35 of the 36 scored answers
  score 0.9 or more and 31 of them are answerable: ECE 0.116 over n = 36, with almost
  every answer in one bin. Isotonic and Platt maps fitted on dev hold one
  unanswerable example and do not fix it (ECE 0.097 and 0.091). The score is the
  scorer model's self-report.
- **The judge is validated on 60 claims by one annotator, and is a Claude model.**
  It is `claude-opus-5-5`, a different model from the agent (`claude-sonnet-4-6`),
  not a different family. κ against the owner's blind labels and the sensitivity on
  perturbed claims are under "Judge validation"; κ on real claims alone rests on one
  human-unsupported claim and is fragile. One annotator gives no inter-annotator
  agreement.
- **n = 56 is small.** The 95 % Wilson intervals in the table show how little it
  pins down: no hallucination in the sample does not rule out a rate up to the
  interval's upper bound.
  Per-claim intervals assume independent claims; claims of one answer are not, so
  those intervals are too narrow. The corpus is 21 fictional documents in three
  domains, with no scanned PDFs or tables.
- **Unanswerable questions were answered** (row "Unanswerable answered"): q-074,
  q-085 and q-089, with scores 0.72, 0.92 and 0.90.

