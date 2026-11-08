# Evaluation methodology

What Verity measures, how each number is computed in the code, and what the numbers
do not show. Where this page and the code disagree, the code is right and this page
is a bug.

## Principles

1. **Every published number comes from an artefact.** The results table in `docs/results.md` is
   generated from `scorecards/live-v1.1.0.json` by
   `scripts/generate_headline_results.py`; CI runs it with `--check`.
2. **CI eval runs on stubs and is a pipeline smoke test.** With `--judge stub` (the
   `--ci` alias) the agent and the judge are scripted stubs. Retrieval metrics are
   real (local models, deterministic); refusal, faithfulness and citation numbers
   describe the stubs, not the system, and are never published as quality metrics.
3. **Refusal is measured, not assumed.** The dataset contains questions the corpus
   cannot answer; the system is scored on refusing them.
4. **Reproducibility rests on the recorded artefact, not on `temperature=0`.** No LLM
   call sends a sampling parameter (Anthropic API `2026-10-21` rejects them). A live
   run is stored once with every answer, citation, judge verdict and token count;
   published numbers come from that artefact, and a rerun is a new sample
   ([ADR 0007](adr/0007-reproducibility-from-artefacts.md)).

## The dataset

`datasets/eval/questions.jsonl` holds 95 questions over the 21 documents of
`datasets/corpus/` (described, with their licence, in `datasets/CORPUS.md`):

| Group | n | Tags |
|---|---:|---|
| answerable | 59 | `single-hop`, `multi-hop` (the evidence spans at least two documents), `negation`, `numeric` |
| unanswerable | 27 | `out-of-corpus`, `near-miss` (adjacent to the corpus, not in it), `false-premise` |
| adversarial | 9 | prompt injection in the question or in a retrieved document; 3 answerable, 6 not |

| Field | Meaning |
|---|---|
| `question` | the query |
| `answerable` | whether the corpus contains the answer |
| `reference_answer` | gold answer, or `null` when `answerable = false` |
| `evidence` | verbatim corpus quotes that answer the question (empty when unanswerable) |
| `relevant_chunk_ids` | gold chunks: every chunk containing one of the `evidence` quotes |
| `tags` | the categories above |

`scripts/build_eval_labels.py` resolves `evidence` to chunk ids by substring match
(never with the retriever it evaluates) and fails if a quote matches no chunk. The
committed ids are the labelling machine's parse. `verity eval run` and `verity eval
retrieval` resolve the gold again, by the same match, against the chunks they ingest,
and exit 2 if a quote is in no chunk: docling's layout model parses the corpus PDFs
differently on other CPUs (on the GitHub runner, numbered headings became list items),
which moves offsets and so chunk ids, not the quotes.

A false-premise question counts as unanswerable: the fact it asks about does not exist
in the corpus, so the expected outcome is a refusal. An answer that corrects the
premise is scored as an answer.

The rule stays, with its cost published. Some answers to false-premise questions say
the premise is false and cite the passage that shows it; the rule scores them as
failures. Why it is kept: a refusal is an event the code records, while "this answer
correctly corrects the premise" needs a grader, and a false-premise question has no
reference answer to grade against. The only grader available is the judge, of the
model family that wrote the questions. Changing the definition after seeing which
answers it penalises would also fit the metric to the run. The cost is published under
"Refusal recall by tag" in [`results.md`](results.md): "unanswerable answered" under
the rule of record, and under a second rule that fails such an answer only when the
judge finds one of its claims unsupported. Which answers pass the second rule is listed
there; that they are correct corrections is a reading of the answers, not a grade.

**Dev / test split.** `scripts/split_dataset.py` draws a stratified split (answerable,
unanswerable, adversarial) with seed `20261102`: `dev.jsonl` (39 questions) and
`test.jsonl` (56). The refusal threshold is tuned on dev only; results are published
on test. Both files are committed and checked in CI, so the split cannot drift.

The v1.0.0 run of record predates this dataset: it used 16 questions over 5 documents
and had no split, so its threshold was tuned and reported on the same examples.

## Metrics as computed

### Retrieval (scored without an LLM)
Computed by `verity.eval.retrieval_metric` on the final fused + reranked list, over
answerable examples only (`relevant_chunk_ids` is empty for out-of-scope ones):
- **precision@k** = |retrieved ∩ gold| / min(k, |retrieved|) (0 when nothing is retrieved);
- **recall@k** = |retrieved ∩ gold| / |gold|;
- **nDCG@k** with binary gain and a log2(rank + 2) discount on 0-based ranks.

Each example has 1-2 gold chunks, so precision@8 cannot exceed 0.25; read it with
recall. Per-stage results (dense, sparse, fused, reranked) are not scored; the
traces record each stage's latency and hit count only. The scoring is deterministic,
but the list it scores is not: it holds the hits retrieved for the sub-questions the
LLM decomposer writes, so a rerun can retrieve different chunks (run-to-run noise
below).

### Answer quality (LLM judge)
The judge (`prompts/judge_faithfulness.prompt`) splits each non-refused answer into
atomic claims and marks each claim `supported` (by the retrieved context) and
`citation_ok` (its citation supports it). Refused answers are not judged.

Per example: faithfulness = supported claims / claims; citation accuracy =
citation_ok claims / claims; hallucination = 1 if the example is answerable and at
least one claim is unsupported, else 0.

Aggregates are defined once, in `verity.eval.answer_quality`, and used by both the
run's scorecard and the published headline:
- **judged** answers are those the judge saw; refusals are in no denominator. An
  answer for which the judge returns no usable claim (unparseable output, or an empty
  claim list) is scored 0 by the judge; it is counted as `n_judged_without_claims`.
- **macro** (per question): mean of the per-answer faithfulness (or citation
  accuracy) over judged answers. A question with 1 claim weighs as much as one with 6.
- **micro** (per claim): supported (or `citation_ok`) claims over all claims, pooled
  across judged answers.
- **hallucination rate**: mean of the per-example value over answerable examples that
  were answered. An answer to an unanswerable question is outside it.
- **invented-answer rate** (the README key result): judged answers with at least one
  unsupported claim (or no usable claim), over every judged answer, answerable or not.
  It counts what a user receives: on test, q-074 (an unanswerable question answered
  with a price of the wrong edition) is in it and not in the hallucination rate. Both
  measure faithfulness to the retrieved passages, not correctness against the
  reference answer.

Both macro and micro are published with their numerator and denominator (questions
or claims). A ratio over an empty set is `null`, never 0 or 1.

### Uncertainty

Every proportion in the scorecard and the results table (per-claim faithfulness and
citation accuracy, hallucination rate, refusal precision and recall, unanswerable
answered) is published as `k/n` with its 95 % Wilson score interval
(`verity.eval.intervals`), computed by `verity.eval.headline` from the artefact.
Wilson rather than the normal approximation because the latter gives [0, 0] at 0/n
and overshoots [0, 1] at the sizes published here. The per-question (macro) figures are
means of per-answer ratios, not proportions, and carry no Wilson interval. The
interval assumes independent trials: questions roughly are, claims of one answer are
not (they share evidence and an author), so per-claim intervals are too narrow.

Two arms run on the same questions are compared by an exact McNemar test on the
questions where they differ (`verity.eval.paired`), not by whether their intervals
overlap: the arms are paired, and two disjoint 95 % intervals is a far stricter
threshold than p < 0.05. Cohen's κ is not a proportion; its interval is a percentile
bootstrap over claims (5000 resamples, seed 1, `scripts/judge_agreement.py kappa-ci`).

Before this definition was shared, the v1.0.0 artefact's scorecard averaged refusals
in at 1.0 (citation accuracy 0.7917 over 16 questions) while the headline excluded
them (0.722 over 12 judged answers). Run artefacts in `datasets/eval/runs/live/` are
immutable (pinned in `SHA256SUMS`); figures recomputed under the current definition
are in `<artefact>.recomputed.json`, written by `scripts/recompute_live_aggregates.py`.

### Run-to-run noise

The agent and the judge sample from LLMs, so a rerun of the same commit on the same
questions does not give the same figures. The range below is how far each figure moved
across reruns of the v1.1.0 dev run; the published figures stay those of the run of
record.

<!-- NOISE:START -->
Commit `6ec544d`, `dev` split (n=39), 3 live runs with the same prompts, models and threshold. The first is the run of record; the others are reruns. Measured on dev; the headline figures are on test, which was not rerun.

| Metric | Run 1 | Run 2 | Run 3 | Min | Max | Range |
|---|---:|---:|---:|---:|---:|---:|
| Faithfulness, per-question mean (judge) | 0.960 | 1.000 | 1.000 | 0.960 | 1.000 | 0.040 |
| Citation accuracy, per-question mean (judge) | 0.939 | 0.973 | 0.944 | 0.939 | 0.973 | 0.034 |
| Hallucination rate (judge) | 0.100 | 0.000 | 0.000 | 0.000 | 0.100 | 0.100 |
| Refusal precision | 0.667 | 0.632 | 0.706 | 0.632 | 0.706 | 0.074 |
| Refusal recall | 0.923 | 0.923 | 0.923 | 0.923 | 0.923 | 0.000 |
| Retrieval nDCG@8 | 0.912 | 0.920 | 0.912 | 0.912 | 0.920 | 0.008 |
| Retrieval recall@8 | 0.968 | 0.987 | 0.968 | 0.968 | 0.987 | 0.019 |
| Retrieval precision@8 | 0.154 | 0.159 | 0.154 | 0.154 | 0.159 | 0.005 |
| Latency p50 (ms, warm) | 4964 | 4841 | 5223 | 4841 | 5223 | 381 |
| Latency p95 (ms, warm) | 8644 | 8265 | 7901 | 7901 | 8644 | 743 |
| Latency p99 (ms, warm) | 10055 | 8795 | 8036 | 8036 | 10055 | 2018 |
| Cold start (ms) | 10178 | 9131 | 8357 | 8357 | 10178 | 1821 |
| Cost per query (USD, agent) | 0.0118 | 0.0116 | 0.0120 | 0.0116 | 0.0120 | 0.0003 |

n=3 runs is a minimum, not a sample size: the range is a lower bound on the run-to-run spread, with no interval. A difference between two versions smaller than this range is not evidence of a change. Generated by `scripts/noise_band.py` from [`datasets/eval/runs/noise/noise_band_6ec544d_dev.json`](../datasets/eval/runs/noise/noise_band_6ec544d_dev.json), kept current by its `--check` in the test suite; the reruns are pinned in `datasets/eval/runs/noise/SHA256SUMS`.
<!-- NOISE:END -->

### Refusal (deterministic over the agent's decisions)
A confusion matrix over `answerable` vs `refused`, computed in `verity.eval.refusal`:
- **refusal recall** = refused out-of-scope / out-of-scope;
- **refusal precision** = refused out-of-scope / refused.

The agent refuses when a deterministic check fails, when the scorer model asks to,
or when the scorer's confidence score is below `confidence_threshold`. The default
threshold is 0.0, the value the dev rule below chose, so in the shipped configuration
it refuses nothing; the cited-relevance check is off by default
(`VERITY_IRRELEVANT_CITATION_CHECK`), on the per-check counts of
[`results.md`](results.md#refusal-checks-one-at-a-time). Each run
records the threshold it used (`EvalRun.confidence_threshold`); the v1.0.0 run
predates that field and used the hand-set default 0.55, on the same examples it
reported.

**Choosing the threshold** (`scripts/refusal_analysis.py`, rules fixed before the
first live run on the dev/test split). The scorer prompt does not contain the
threshold, so the refusals at any threshold are recomputed from the scores recorded
in the dev run artefact (`verity.eval.calibration`):
1. sweep thresholds 0.00, 0.05, ..., 1.00 on dev;
2. take the highest refusal recall among thresholds with refusal precision ≥ 0.9
   (at most one refusal in ten hits an answerable question); ties go to the lowest
   threshold;
3. if no threshold reaches 0.9, take the highest refusal F1 instead; the report
   names the rule that applied;
4. run the test split once at that threshold and publish its numbers.

**Calibration.** The confidence score is read as the probability that the question is
answerable, over the examples the scorer scored (a deterministic refusal has no
score). ECE uses 10 equal-width bins: Σ |B|/n · |mean score(B) − answerable
fraction(B)|. Isotonic and Platt maps are fitted on dev and their ECE measured on
test; they are reported, not used by the agent. The reliability diagram plots the
bins with their sizes. "Calibrated" is claimed only if the test ECE supports it.

**What v1.1.0 measured** (`scorecards/refusal-analysis.json`,
`docs/figures/reliability.png`; the system did not change between the dev run at
`6ec544d` and the test run at `969e1d5`, which only add artefacts):
- On dev (n = 39), refusal precision 0.667 and recall 0.923 at every threshold from
  0.00 to 0.95: every refusal had a deterministic cause or was the scorer's own, and
  every scored answer scored ≥ 0.9. No threshold reached precision 0.9, so the
  fallback rule applied and gave 0.00.
- On test (n = 56) at 0.00: precision 0.773 (17 / 22), recall 0.850 (17 / 20).
  No refusal has the cause `below_threshold`; the 5 refusals of answerable questions
  are 4 `irrelevant_citation` and 1 `low_citation_coverage`.
- Test ECE 0.116 over the 36 scored answers, 35 of them in the top bin. Isotonic and
  Platt maps fitted on dev, whose scored answers hold a single unanswerable question,
  give 0.097 and 0.091. The score is not calibrated and the README does not claim it.
- Improving refusal precision means changing the citation checks, not the threshold.
  That tuning uses dev only; a new test run belongs to a new version.

### Operational
- **Latency** p50 / p95 / p99 of the end-to-end `agent.answer` wall clock. With
  `--no-warmup` (the v1.0.0 run) the first question's model loading is included and
  `cold_start` is `null`; with warmup, cold start is measured separately.
- **Cost / query** in USD from token counts × the per-million-token list prices in
  `DEFAULT_MODEL_PRICES` (`verity.config`), overridable with `VERITY_MODEL_PRICES`; not
  from provider invoices. Averaged over every attempted question: an example that ends
  in an infrastructure error is kept out of the quality metrics, but the calls it made
  before the error were billed, so they count in the cost and in `--budget-usd`.

## Judge reliability

The judge is an LLM and a source of error. In the v1.0.0 run the judge and the agent
are the same model (`claude-sonnet-4-6`), and judge–human agreement has not been
measured, so the answer-quality metrics are **unvalidated**. `Settings.judge_model` existed
then but nothing read it: the live judge was built from the agent's own client.

The judge is now chosen by `create_judge_client` (`verity.llm.routing`): another model
family when its key is set (`VERITY_JUDGE_PROVIDER=auto`, OpenAI), otherwise a Claude
model different from the agent's (`VERITY_JUDGE_ANTHROPIC_MODEL`, default
`claude-opus-5-5`). It has a single client and no fallback, so the model named in the
scorecard is the one that judged; a judge equal to the agent's model stops the run.
Whatever the judge, Cohen's κ against human labels is required before its numbers are
published as validated; until then the results table marks them "judge unvalidated", and
`scripts/generate_headline_results.py` names κ only for the model it was measured on. κ alone
cannot validate this judge: it marked 161 of the 164 live claims supported, so on a
sample of real claims its labels are almost all "supported" and κ is undefined or
rests on one or two items. The
validation (`scripts/judge_agreement.py`) therefore mixes 40 real claims, the 3
unsupported ones included, with 20 claims perturbed to be false by
`verity.eval.perturb` (one figure, date, negation or same-type entity changed, five
of each). Every answer gives two items, both real or both perturbed, so a repeated
block of extracts says nothing about origin. A human labels all 60 blind, without
origin or verdict, in two files of 30. Published separately: the judge's sensitivity
(perturbed claims it finds unsupported, also among those the human finds
unsupported, since a perturbation can leave a claim true) and κ on the mixed set.
The two halves are judged under two protocols: real claims carry the verdict of the
live run, where the judge split the whole answer; a perturbed claim is judged alone
and counts as detected if any sub-claim the judge makes of it is unsupported, so
sensitivity is an upper bound. κ on real claims only and the judge's specificity on
them are published next to the mixed κ. κ is not a proportion and has no Wilson
interval. The owner labelled the 60 claims blind in one pass, with a terminal tool
that reads only the annotation CSVs; those labels are the
reference and are published as labelled, pinned with the key in
`datasets/eval/judge_validation/SHA256SUMS`. κ on real claims alone is marked fragile
when the smaller human class holds fewer than five claims. The disagreements, and
the perturbed claims the annotator judged true, are adjudicated afterwards with the
judge's verdict visible; that adjudication is published in its own files and its own
section of `docs/results.md` and never rewrites the blind labels or `agreement.json`. One annotator gives
no inter-annotator agreement. v1.1.0
was judged by `claude-opus-5-5` (no OpenAI key was available), a different Claude
model from the agent, not a different family. Claim-level verdicts are
stored per example in the run artefact so disagreements can be inspected.

## Regression tracking

Each run is saved as an `EvalRun` JSON (scorecard + git SHA + agent, judge and
embedding model ids) under `datasets/eval/runs/` (live runs under `runs/live/`).
`verity eval compare` diffs recent runs whose provenance matches.

## What would make a number untrustworthy

- a scorecard without a git SHA;
- a headline number computed on stub output (CI mode);
- answer-quality numbers without a judge–human agreement figure (true of v1.0.0);
- refusal recall reported without refusal precision;
- a threshold tuned and reported on the same examples (true of v1.0.0, which predates
  the dev/test split);
- eval examples leaking into retriever or prompt tuning.
