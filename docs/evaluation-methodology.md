# Evaluation methodology

What Verity measures, how each number is computed in the code, and what the numbers
do not show. Where this page and the code disagree, the code is right and this page
is a bug.

## Principles

1. **Every published number comes from an artefact.** The README results table is
   generated from `scorecards/live-v1.0.0.json` by
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
(never with the retriever it evaluates) and fails if a quote matches no chunk.

A false-premise question counts as unanswerable: the fact it asks about does not exist
in the corpus, so the expected outcome is a refusal. An answer that corrects the
premise is scored as an answer.

**Dev / test split.** `scripts/split_dataset.py` draws a stratified split (answerable,
unanswerable, adversarial) with seed `20261102`: `dev.jsonl` (39 questions) and
`test.jsonl` (56). The refusal threshold is tuned on dev only; results are published
on test. Both files are committed and checked in CI, so the split cannot drift.

The v1.0.0 run of record predates this dataset: it used 16 questions over 5 documents
and had no split, so its threshold was tuned and reported on the same examples.

## Metrics as computed

### Retrieval (deterministic, no LLM)
Computed by `verity.eval.retrieval_metric` on the final fused + reranked list, over
answerable examples only (`relevant_chunk_ids` is empty for out-of-scope ones):
- **precision@k** = |retrieved ∩ gold| / min(k, |retrieved|) (0 when nothing is retrieved);
- **recall@k** = |retrieved ∩ gold| / |gold|;
- **nDCG@k** with binary gain and a log2(rank + 2) discount on 0-based ranks.

Each example has 1-2 gold chunks, so precision@8 cannot exceed 0.25; read it with
recall. Per-stage results (dense, sparse, fused, reranked) are not scored; the
traces record each stage's latency and hit count only.

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
  were answered.

Both macro and micro are published with their numerator and denominator (questions
or claims). A ratio over an empty set is `null`, never 0 or 1.

Before this definition was shared, the v1.0.0 artefact's scorecard averaged refusals
in at 1.0 (citation accuracy 0.7917 over 16 questions) while the headline excluded
them (0.722 over 12 judged answers).

### Refusal (deterministic over the agent's decisions)
A confusion matrix over `answerable` vs `refused`, computed in `verity.eval.refusal`:
- **refusal recall** = refused out-of-scope / out-of-scope;
- **refusal precision** = refused out-of-scope / refused.

The agent refuses when a deterministic check fails, when the scorer model asks to,
or when the scorer's confidence score is below `confidence_threshold`. Each run
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

### Operational
- **Latency** p50 / p95 / p99 of the end-to-end `agent.answer` wall clock. With
  `--no-warmup` (the v1.0.0 run) the first question's model loading is included and
  `cold_start` is `null`; with warmup, cold start is measured separately.
- **Cost / query** in USD from token counts × the per-1k prices set in
  `verity.llm.clients`, averaged over every attempted question. An example that ends
  in an infrastructure error is kept out of the quality metrics, but the calls it made
  before the error were billed: they count in the cost and in `--budget-usd`.

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
published. Claim-level verdicts are
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
