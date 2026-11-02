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

`datasets/eval/questions.jsonl`, 16 examples over the 5 documents in
`datasets/corpus/`: 11 answerable, 5 out-of-scope. Each example has:

| Field | Meaning |
|---|---|
| `question` | the query |
| `answerable` | whether the corpus contains the answer |
| `reference_answer` | gold answer, or `null` when `answerable = false` |
| `relevant_chunk_ids` | gold chunk ids (filled for all 11 answerable examples) |
| `tags` | `single-hop`, `multi-hop`, `numeric`, `near-miss`, `out-of-scope` |

16 examples support a method check, not a benchmark: one example moves refusal
recall by 0.2.

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

Published aggregate (`verity.eval.headline`, schema v3):
- **faithfulness** and **citation accuracy**: mean of the per-example values over
  judged answers (scope `answered`). This is a **macro** (per-question) average: a
  question with 1 claim weighs as much as one with 6. The per-claim (**micro**)
  aggregate is not published yet; it differs (see the README's open issues).
- **hallucination rate**: mean over answerable examples that were answered (scope
  `answerable_and_answered`).

Each metric is stored with its `scope` and `n_denominator`.

### Refusal (deterministic over the agent's decisions)
A confusion matrix over `answerable` vs `refused`, computed in `verity.eval.refusal`:
- **refusal recall** = refused out-of-scope / out-of-scope;
- **refusal precision** = refused out-of-scope / refused.

The agent refuses when the scorer model asks to, or when its self-reported
confidence is below `confidence_threshold` (default 0.55, the value at the v1.0.0
run's commit; the artefact does not record it). The threshold was set by hand,
not tuned on a dev split, and the confidence score's calibration has not been
measured. The scorecard JSON names this block `refusal_calibration`; the name
predates that measurement and does not claim it.

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
- a threshold tuned and reported on the same examples (true of v1.0.0: there is no
  dev/test split);
- eval examples leaking into retriever or prompt tuning.
