# Results

The detail behind the README's key result: the full results table of the run of
record, the judge's validation against blind human labels, and what the numbers do not say.

## Measured results

<!-- RESULTS:START -->
**Measured on**: `agent=claude-sonnet-4-6 (54/54)` · `judge=claude-opus-5-5 (40/54 answered — 14 refusals bypass the judge)` · `embedding=BAAI/bge-small-en-v1.5` · commit `513b839` · 2026-11-07 · dataset `test` · n=56 (36 answerable + 18 unanswerable) · **0 errored, 2 not run** (kept out of every metric) · total cost **$2.4562**

| Metric | Value | Notes |
|---|---:|---|
| **Invented answers** | **1/40** | ↓ better · answers with a claim the judge finds unsupported (or no usable claim) / every judged answer, answerable or not: 1/40 · 95 % CI [0.004, 0.129] · `q-089` · judge prompt v0.2.0 κ 0.60 vs blind human on 52 of 60 claims (its rejudge stopped at its spending cap; no bootstrap interval) |
| **Refusal recall** | **0.722** | unanswerable questions refused / unanswerable: 13/18 · 95 % CI [0.491, 0.875] · threshold 0 |
| **Refusal precision** | **0.929** | refusals that were unanswerable / refusals: 13/14 · 95 % CI [0.685, 0.987] |
| Faithfulness, per question | 0.996 | mean over 40 judged answers · judge prompt v0.2.0 κ 0.60 vs blind human on 52 of 60 claims (its rejudge stopped at its spending cap; no bootstrap interval) |
| Faithfulness, per claim | 0.992 | pooled over claims: 121/122 · 95 % CI [0.955, 0.999] · judge prompt v0.2.0 κ 0.60 vs blind human on 52 of 60 claims (its rejudge stopped at its spending cap; no bootstrap interval) |
| Citation accuracy, per question | 0.962 | mean over 40 judged answers · judge prompt v0.2.0 κ 0.60 vs blind human on 52 of 60 claims (its rejudge stopped at its spending cap; no bootstrap interval) |
| Citation accuracy, per claim | 0.951 | pooled over claims: 116/122 · 95 % CI [0.897, 0.977] · judge prompt v0.2.0 κ 0.60 vs blind human on 52 of 60 claims (its rejudge stopped at its spending cap; no bootstrap interval) |
| Hallucination rate, answerable only | 0.000 | ↓ better · answers with an unsupported claim / answered answerable questions (an answer to an unanswerable question is outside it): 0/35 · 95 % CI [0.000, 0.099] |
| Unanswerable answered | 5/18 | ↓ better · an answer shipped where the corpus has none: 5/18 · 95 % CI [0.125, 0.509] |
| Retrieval nDCG@8 | 0.923 | scored without an LLM on hits retrieved for the LLM sub-questions |
| Retrieval recall@8 | 0.972 | as nDCG |
| Retrieval precision@8 | 0.142 | k>gold ceiling; interpret with recall |
| Latency p50 / p95 / p99 | 6538.4 ms / 9857.9 ms / 11140.9 ms | warm; cold start `8884.6 ms` measured separately |
| Cost / query | $0.0235 | $0.0235 agent only · $0.0455 with the judge · total $2.4562 = agent $1.2671 + judge $1.1890 |

Refusals by cause: `low_citation_coverage` 7 (1 answerable) · `no_citation` 4 (0 answerable) · `suspected_injection` 3 (0 answerable).
No refusal has the cause `below_threshold`: the confidence threshold decided none of them.

Generated from [`scorecards/live-v1.1.0.json`](../scorecards/live-v1.1.0.json), itself computed from the run artefact `datasets/eval/runs/live/eval_513b839_test.incomplete-20261107T122042441670Z.json` (`scripts/generate_headline_results.py --check` recomputes both).

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
[`datasets/eval/runs/live/eval_513b839_test.incomplete-20261107T122042441670Z.json`](../datasets/eval/runs/live/eval_513b839_test.incomplete-20261107T122042441670Z.json).
The run used the in-memory store (BM25 sparse arm); `verity eval run` defaults to
pgvector (`--store memory` reproduces it). Definitions:
[`docs/evaluation-methodology.md`](evaluation-methodology.md).
Each proportion with its k/n and Wilson 95 % interval, the approved SLOs and whether
this run meets them, the intended use and the known limits:
[`docs/SYSTEM_CARD.md`](SYSTEM_CARD.md), generated from the same scorecard.

### Refusal checks, one at a time

Every refusal of an answerable question comes from the deterministic checks. The
tables below say which check makes which refusals, without a model call.

<!-- GATE-ABLATION:START -->
Right / wrong refusals: a refusal of an unanswerable question is right, of an answerable one wrong.

**First failing check, on Verity's runs.** A refused record keeps only the first check that failed, so a later check is never seen on it. Each cell is records, then distinct questions in brackets: the dev runs include reruns of the same questions.

| Check | dev (140 records) | test (56 records) |
|---|---|---|
| `irrelevant_citation` | 5 / 11 (2 / 4) | 0 / 4 (0 / 4) |
| `low_citation_coverage` | 13 / 12 (5 / 4) | 6 / 1 (6 / 1) |
| `model_refused` | 0 / 0 (0 / 0) | 2 / 0 (2 / 0) |
| `no_citation` | 13 / 0 (4 / 0) | 6 / 0 (6 / 0) |
| `suspected_injection` | 7 / 0 (2 / 0) | 3 / 0 (3 / 0) |

Runs: dev: `eval_6ec544d_dev.json`, `eval_6ec544d_dev.rerun2.json`, `eval_6ec544d_dev.rerun3.json`, `eval_e644a98_dev.incomplete-20261105T094431636600Z.json`; test: `eval_969e1d5_test.json`.

With `irrelevant_citation` off, refusal precision on dev lies between 33/56 and 38/50 (records): each record it refused becomes an answer or a scorer refusal.

With `irrelevant_citation` off, refusal precision on test lies between 17/22 and 17/18 (records): each record it refused becomes an answer or a scorer refusal.

**Every check on its own, on the naive arm's dev drafts** (28 drafts, 24 answerable; thresholds: coverage 0.5, cited reranker score 0.0). The naive arm ran the same pipeline and shipped its drafts unchecked, so each check can be replayed on every draft. A draft with no citation has coverage 0, so `low_citation_coverage` alone also counts the `no_citation` drafts.

| Check | Alone: right / wrong | All checks but this one: right / wrong | Wrongly refused alone |
|---|---|---|---|
| `no_citation` | 3 / 0 | 4 / 5 | none |
| `suspected_injection` | 1 / 0 | 3 / 5 | none |
| `low_citation_coverage` | 3 / 2 | 4 / 3 | `q-022`, `q-031` |
| `irrelevant_citation` | 0 / 3 | 4 / 2 | `q-001`, `q-023`, `q-063` |
| all four | 4 / 5 | — | — |

The replay is checked in one direction: on every answered record of the Verity runs above, it finds that every check passes, as the agent did. It cannot check a refused record, which keeps no draft. Generated from `scorecards/gate-ablation.json` (`scripts/gate_ablation.py --check`).
<!-- GATE-ABLATION:END -->

**Decision, on dev only.** The cited-relevance check (`irrelevant_citation`) makes more
wrong refusals than right ones in both dev readings, so it is off by default
(`VERITY_IRRELEVANT_CITATION_CHECK=true` restores it). The dev evidence is small: as
the first failing check it wrongly refused 4 distinct questions and rightly refused 2;
alone on the naive drafts it refused 3 answerable drafts and no unanswerable one, and
those three are among the four. The cost is the 2 unanswerable dev questions it was
the first to refuse (`q-072`, `q-088`, see `right_ids` in the JSON): with the check off
their drafts reach the scorer and can ship, which the replay cannot show, since the
naive drafts hold no unanswerable question this check refused. The precision ranges
above bound the effect on the runs of record.

Scope: the tables above replay the runs made with the check on (`6ec544d` dev,
`969e1d5` test). The headline in the README and the results table come from the test
run `513b839`, made after the decision with the check off, the shipped default. Test
figures were not used for the decision.

### Verity against a naive RAG baseline, 95 questions

<!-- TWO-ARM:START -->
93 questions completed by both arms (dev + test), agent `claude-sonnet-4-6`, judge `claude-opus-5-5`, commit `513b839`. [Pre-registered](preregistration/two-arm-comparison.md) before the runs: a self-abstention counts as a refusal; exact McNemar on the paired questions, Holm-adjusted over the three primary tests.

| Metric | Better | Verity | Naive RAG | Verity only / naive only | p (Holm) | Reading |
|---|---|---|---|---|---|---|
| **Unanswerable answered** | lower | 4/31 [0.051, 0.288] | 12/31 [0.237, 0.562] | 0 / 8 | 0.0078 (0.0234) | Verity's checks refuse unanswerable questions that the naive draft answers. |
| **Answerable answered** | higher | 58/62 [0.846, 0.975] | 62/62 [0.942, 1.000] | 0 / 4 | 0.1250 (0.2500) | No cost measurable at this n. |
| **Invented answer** | lower | 0/93 [0.000, 0.040] | 3/93 [0.011, 0.091] | 0 / 3 | 0.2500 (0.2500) | No difference in invented answers measurable at this n. |
| Unanswerable answered, self-abstentions as answers | — | 7/31 [0.114, 0.398] | 31/31 [0.890, 1.000] | 0 / 24 | < 0.0001 | descriptive |
| Answerable answered, self-abstentions as answers | — | 58/62 [0.846, 0.975] | 62/62 [0.942, 1.000] | 0 / 4 | 0.1250 | descriptive |
| Invented answer, self-abstentions as answers | — | 1/93 [0.002, 0.058] | 5/93 [0.023, 0.120] | 0 / 4 | 0.1250 | descriptive |
| Invented, over the answers each arm shipped | — | 1/65 [0.003, 0.082] | 5/93 [0.023, 0.120] | — | — | not paired |
| Hallucination rate, answerable questions answered | — | 0/58 [0.000, 0.062] | 2/62 [0.009, 0.110] | — | — | not paired |

Unanswerable answered by tag (self-abstentions as refusals; a question can carry several tags), descriptive:

| Tag | Verity | Naive RAG |
|---|---|---|
| `adversarial` | 0/4 [0.000, 0.490] | 3/4 [0.301, 0.954] |
| `false-premise` | 2/7 [0.082, 0.641] | 3/7 [0.158, 0.750] |
| `near-miss` | 2/15 [0.037, 0.379] | 4/15 [0.109, 0.519] |
| `out-of-corpus` | 0/9 [0.000, 0.299] | 5/9 [0.267, 0.811] |

- Verity: $2.1055 agent, $1.8621 judge; per query $0.0226 agent only, $0.0427 with the judge. `eval_513b839_test.incomplete-20261107T122042441670Z.json` stopped after 54 (budget_exceeded).
- Naive RAG: $1.6349 agent, $2.6551 judge; per query $0.0172 agent only, $0.0452 with the judge.
- Completed by one arm only, left out of every row: `q-094`, `q-095`.
- Discordant questions, unanswerable answered: Verity only none; naive only `q-004`, `q-006`, `q-015`, `q-017`, `q-018`, `q-019`, `q-080`, `q-084`.

Generated from `scorecards/two-arm-comparison.json`, computed from the four artefacts it names (`scripts/compare_two_arm.py --check`).
<!-- TWO-ARM:END -->

**What it implies for the design.** The refusal checks do what they are built for:
they refuse unanswerable questions that the synthesizer's own draft answers, and the
table by tag shows where (out-of-corpus and adversarial questions). Their cost on
answerable questions is not measurable at this n: every discordant answerable question
went to the naive arm, but too few of them to reject (power for a 10-point cost at the
62 pairs: 0.427). The difference in invented answers is below what 93 questions can
show (power 0.184 at 93 pairs for 5 points; the pre-registration computed 0.198 at 95).
Half the effect on unanswerable questions comes from the injection-marker check: of
the 8 questions only the naive arm answered, Verity refused `q-004`, `q-017`, `q-018`
and `q-019` for `suspected_injection`, `q-006` for `no_citation`, and `q-015`, `q-080`
and `q-084` for `low_citation_coverage`. Over the answers each arm shipped, Verity
invents 1/65 against 0/93 in the primary row: the one is `q-089`, a self-abstention
that also states a claim the judge finds unsupported; the self-abstention rule counts
it as a refusal, and the row "Invented answer, self-abstentions as answers" counts it.
Deviations from the pre-registration, found after the runs, are listed at its end. Scope: two arms on the shipped defaults. Known
limit: the full-corpus arm, the obvious baseline for a 5,489-word corpus, is not run;
its measured cost is in the pre-registration. Next step: that arm, with prompt caching
or its own budget.

### Correctness against the reference answer

The rows above measure faithfulness: whether the judge finds each claim supported by
the passages retrieved for it. An answer can be faithful to the wrong passage. Here,
each answered answerable question is graded against the dataset's `reference_answer`.

<!-- CORRECTNESS:START -->
| Run | Correct | Partial | Incorrect | Unparsed | Not correct | Numbers in the reference found in the answer |
|---|---|---|---|---|---|---|
| `datasets/eval/runs/live/eval_513b839_dev-naive-baseline.json` (26) | 23/26 · [0.710, 0.960] | 2 | 1 | 0 | `q-001`, `q-002`, `q-023` | 15/20; disagrees with the grade on `q-011`, `q-012`, `q-057` |
| `datasets/eval/runs/live/eval_513b839_dev.json` (23) | 20/23 · [0.679, 0.955] | 2 | 1 | 0 | `q-001`, `q-002`, `q-023` | 13/18; disagrees with the grade on `q-011`, `q-012`, `q-057` |
| `datasets/eval/runs/live/eval_513b839_test-naive-baseline.json` (36) | 36/36 · [0.904, 1.000] | 0 | 0 | 0 | none | 26/30; disagrees with the grade on `q-029`, `q-034`, `q-055`, `q-062` |
| `datasets/eval/runs/live/eval_513b839_test.incomplete-20261107T122042441670Z.json` (35) | 35/35 · [0.901, 1.000] | 0 | 0 | 0 | none | 24/29; disagrees with the grade on `q-029`, `q-032`, `q-034`, `q-058`, `q-062` |
| `datasets/eval/runs/live/eval_6ec544d_dev.json` (20) | 19/20 · [0.764, 0.991] | 1 | 0 | 0 | `q-001` | 11/16; disagrees with the grade on `q-002`, `q-012`, `q-057`, `q-060` |
| `datasets/eval/runs/live/eval_969e1d5_test.json` (31) | 31/31 · [0.890, 1.000] | 0 | 0 | 0 | none | 21/25; disagrees with the grade on `q-029`, `q-032`, `q-034`, `q-062` |

Judge `claude-opus-5-5` with `prompts/judge_correctness.prompt`, total $0.6356. Scope: the answerable questions each run answered, graded against the dataset's `reference_answer`; the answers are the artefact's, the agent was not called again. The grade is **not validated against human labels**; the number check is deterministic but covers only references with a number. Generated from `scorecards/correctness.json` (`scripts/judge_correctness.py render --check`); verdicts and reasons in `datasets/eval/correctness/`.
<!-- CORRECTNESS:END -->

The number check and the grade disagree only where the check is too literal. That reading
was made by the coding agent that prepared this page, not a human label: in each
disagreement, the number missing from the answer is a section number, a date, a detail
the question does not ask for, or "one" used as a pronoun.

### Refusal recall by tag

The aggregate refusal recall above counts `out-of-corpus` questions (a football result,
a drug dose, asked of a company corpus), the easiest to decline, and `adversarial` ones, whose refusals come from the injection-marker check or
from a draft with no citation (causes under "Refusals by cause"). The rows below leave
them out, so the recall on the questions that test the design (`near-miss`,
`false-premise`) stands on its own n.

<!-- REFUSAL-BY-TAG:START -->
Run `datasets/eval/runs/live/eval_513b839_test.incomplete-20261107T122042441670Z.json`, tags from `datasets/eval/test.jsonl`. Refusal recall = unanswerable questions refused / unanswerable questions, Wilson 95 % interval. A question can carry several tags, so tag rows overlap.

| Unanswerable questions | Refused | Answered |
|---|---|---|
| all | 13/18 · [0.491, 0.875] | `q-074`, `q-078`, `q-081`, `q-085`, `q-089` |
| without `out-of-corpus` | 8/13 · [0.355, 0.823] | `q-074`, `q-078`, `q-081`, `q-085`, `q-089` |
| without `out-of-corpus` and `adversarial` | 8/13 · [0.355, 0.823] | `q-074`, `q-078`, `q-081`, `q-085`, `q-089` |
| tag `adversarial` | 2/2 · [0.342, 1.000] | none |
| tag `false-premise` | 1/3 · [0.061, 0.792] | `q-085`, `q-089` |
| tag `near-miss` | 7/10 · [0.397, 0.892] | `q-074`, `q-078`, `q-081` |
| tag `out-of-corpus` | 5/5 · [0.566, 1.000] | none |

Answerable questions refused, by tag: `adversarial` 0/1 · [0.000, 0.793] · `multi-hop` 1/6 · [0.030, 0.564] · `negation` 0/9 · [0.000, 0.299] · `numeric` 0/11 · [0.000, 0.259] · `single-hop` 0/11 · [0.000, 0.259].

Unanswerable answered: 5/18 · [0.125, 0.509] under the rule of record (any answer to an unanswerable question fails). If an answer failed only when the judge finds one of its claims unsupported, it would be 1/18 · [0.010, 0.258]: `q-089` invented; `q-074`, `q-078`, `q-081`, `q-085` fully supported.

Generated from `scorecards/refusal-by-tag.json` (`scripts/refusal_by_tag.py --check`).
<!-- REFUSAL-BY-TAG:END -->

## Judge validation

The answer-quality rows above are judged by a model. Its agreement with blind human
labels is the result of record; an adjudication of the disagreements, published
separately, does not replace it.

<!-- JUDGE:START -->
Judge `claude-opus-5-5` with judge prompt v0.1.0 (`datasets/eval/judge_validation/judge_faithfulness.v0.1.0.prompt`, sha256 `2c5f8eca49ba`) against **blind human labels**, the reference: 60 claims labelled without origin or judge verdict, 40 real claims (16 from the dev run, 24 from the test run) and 20 claims perturbed to be false (one number, date, negation or entity changed). The κ below belongs to this prompt version: a changed judge prompt is a new judge and needs its own κ.

The judge is validated on 60 claims, of which 1 real hallucination (a real claim the human finds unsupported). On the real claims the judge flags 3 unsupported and the human confirms 1. The perturbed claims are a proxy for its sensitivity to real hallucinations, not a bound on it.

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

- **The run of record is incomplete.** It stopped at its $2.40 spending cap after
  54 of 56 questions; `q-094` and `q-095`, both unanswerable and tagged
  `adversarial`, were not run and are outside every metric. Scope: 54 questions, 36
  answerable and 18 unanswerable. Known limit: the two adversarial questions most
  likely to be refused by `suspected_injection` are missing. Next step: the next test
  run with a cap sized on the judged-answer cost.
- **Several analyses below rest on the earlier runs.** The threshold sweep, the
  calibration figures and the judge validation of prompt v0.1.0 come from the dev run
  `6ec544d` and the test run `969e1d5`, which used `synthesize.prompt` v0.2.0 and
  judge prompt v0.1.0. Each bullet names the run it rests on.
- **The threshold decides nothing on this data, and refuses nothing by default.**
  The shipped default is 0.0, the value the dev rule chose, which no score in
  [0, 1] is below; the run of record used it. Every refusal above has a deterministic
  cause (no citation, low citation coverage, suspected injection); none is
  `below_threshold`.
  The threshold was chosen on the dev split by a rule fixed before the run (highest
  refusal recall at precision ≥ 0.9, else highest F1): no threshold reached 0.9,
  refusal precision and recall are identical for every threshold from 0.00 to 0.95,
  and the rule gave 0.00. Sweep and figure:
  [`scorecards/refusal-analysis.json`](../scorecards/refusal-analysis.json),
  [`docs/figures/reliability.png`](figures/reliability.png).
- **Refusal precision is a known limit** (row above, with its interval). On the run
  of record the one refusal of an answerable question, `q-055`, comes from
  `low_citation_coverage` (counts under "Refusals by cause"); on dev the citation
  checks are the only cause too (`scorecards/refusal-analysis.json`). Tuning those checks
  is done on dev only; a new test run belongs to a new version.
- **The confidence score is not calibrated.** On the test run `969e1d5`, 35 of the 36 scored answers
  score 0.9 or more and 31 of them are answerable: ECE 0.116 over n = 36, with almost
  every answer in one bin. Isotonic and Platt maps fitted on dev hold one
  unanswerable example and do not fix it (ECE 0.097 and 0.091). The score is the
  scorer model's self-report.
- **Most of the corpus and the questions were written with Claude Opus 5.5**, the
  judge's model family, by the author of the system (16 of 21 documents and the 95
  questions; [origin](../datasets/CORPUS.md)). Scope: 21 fictional documents. Known limit: self-preference of the judge
  and questions shaped by the system's author are not measured. Next step: a public
  corpus with third-party questions.
- **The judge is validated on 60 claims by one annotator, and is a Claude model.**
  It is `claude-opus-5-5`, a different model from the agent (`claude-sonnet-4-6`),
  not a different family. κ against the owner's blind labels and the sensitivity on
  perturbed claims are under "Judge validation"; κ on real claims alone rests on one
  human-unsupported claim (q-074's price) and is fragile: its bootstrap interval is
  [0.00, 1.00]. One annotator gives no inter-annotator
  agreement.
- **n = 54 is small.** The 95 % Wilson intervals in the table show how little it
  pins down: no hallucination in the sample does not rule out a rate up to the
  interval's upper bound.
  Per-claim intervals assume independent claims; claims of one answer are not, so
  those intervals are too narrow. The corpus is 21 fictional documents in three
  domains, with no scanned PDFs or tables.
- **Unanswerable questions were answered** (row "Unanswerable answered"): q-074,
  q-078, q-081, q-085 and q-089, with scores 0.55, 0.97, 0.95, 0.95 and 0.82.

