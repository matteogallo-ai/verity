# Pre-registration: Verity against a naive RAG baseline

Committed before either run, and before the code that runs the baseline. The metrics,
the reading rule and the arms below are fixed by this file; the result is published
as it comes out, favourable or not.

## Question

Same questions, same model, same retrieval, same synthesis prompt: what do Verity's
checks change compared with shipping the synthesizer's draft as it is?

## Data and models

- Dataset: `datasets/eval/dev.jsonl`, sha256
  `c922d7a680b1d647eaef5e22beb4d3c314c3e983ee7c1b634bd261ebb1e6809f`: 39 questions,
  26 answerable, 13 unanswerable. The test split is not used.
- Agent model: `claude-sonnet-4-6` for both arms (decompose, synthesize, and in the
  Verity arm the scorer).
- Judge: `claude-opus-5-5` with `prompts/judge_faithfulness.prompt` v0.2.0 (sha256
  `aa66e62370a2…`), the same for both arms. Its κ against the blind human labels is
  0.60 on the 52 claims it judged (`docs/results.md`, "Judge prompt v0.2.0").
- Store: in-memory (`--store memory`), as the runs of record.
- Prompts: `decompose` v0.2.0, `synthesize` v0.3.0, `confidence` v0.2.0, as in the
  repository at the commit of the runs.

## Arms

Both arms run at the same commit, one after the other, once each.

- **Verity**: the pipeline as shipped, at the confidence threshold of record,
  `VERITY_CONFIDENCE_THRESHOLD=0.0` (chosen on dev for v1.1.0). A draft is refused when
  it has no verified citation, when the support check fails (citation coverage below
  0.5, a cited chunk the reranker scores below 0.0, injection markers), when the scorer
  refuses, or when its score is below the threshold.
- **Naive RAG**: the same decomposition, retrieval and synthesis; the draft ships as it
  is. No no-citation refusal, no support check, no scorer call, no threshold. The
  synthesizer still drops a citation whose quote is not in the cited chunk: a
  `Citation` must point at a document span to exist at all. That filter decides which
  citations are attached, never whether an answer ships. The naive arm records no
  refusal; an empty draft (synthesizer output that does not parse) ships as an empty
  answer and counts as answered.

## Metrics compared

Each is a proportion k/n, reported with its Wilson 95 % interval per arm.

Primary:

1. **Unanswerable answered**: unanswerable questions that got an answer / 13. Lower is
   better.
2. **Hallucination rate**: answered answerable questions with at least one claim the
   judge finds unsupported / answered answerable questions. Lower is better.
3. **Answerable answered**: answerable questions that got an answer / 26. Higher is
   better; it is what the checks cost.

Secondary:

4. Unanswerable answered, **self-abstentions counted as refusals**: an answer whose text
   matches one of the patterns below (case-insensitive) is counted as a refusal in
   either arm. This corrects metric 1 for a naive draft that itself says the documents
   do not answer.

   ```
   \b(do|does|did) not (contain|mention|provide|include|specify|say|state|address|cover|describe)\b
   \b(don't|doesn't) (contain|mention|provide|include|specify|say|state|address|cover|describe)\b
   \bno (information|mention|details?|data)\b
   \b(is|are|was|were) not (mentioned|specified|stated|provided|covered|available|included|described)\b
   \b(cannot|can't|unable to) (be )?(determine|answer|find|confirm)
   \bI don't know\b
   ```

5. Faithfulness, per claim: supported claims / claims, pooled over judged answers.
6. Citation accuracy, per claim: claims whose citation the judge finds supporting /
   claims.

Per-claim intervals assume independent claims; claims of one answer are not
independent, so those two intervals are too narrow.

## Reading rule

For each metric, compare the two Wilson 95 % intervals:

- the intervals overlap → **"not distinguishable"** on this sample;
- they do not overlap → **"distinguishable"**, with the direction stated.

Nothing else is tested; no p-value, no pooling over metrics, no re-run. With n = 13
unanswerable and 26 answerable questions, overlapping intervals are the expected
outcome for any but a large difference, and are published as such.

## Budget and execution

- Hard caps coded in `scripts/run_naive_baseline.sh`: Verity arm $0.92, naive arm
  $1.22. The harness checks its cap after each call, so an arm can pass it by one
  call (about $0.03); the $0.06 left keeps the total under the $2.20 the owner
  approved. Estimate: Verity ≈ $0.85, naive ≈ $1.06.
- Amended before either run: the caps first read $0.95 and $1.25, which summed to the
  approved $2.20 with no room for that last call.
- An arm that hits its cap or has errored questions is published incomplete, with the
  questions it did not complete named; the comparison then runs on the questions both
  arms completed, and says so.
- Artefacts: `datasets/eval/runs/live/eval_<sha>_dev.json` (Verity) and
  `eval_<sha>_dev-naive-baseline.json` (naive), pinned in that directory's
  `SHA256SUMS`, never edited.
- The comparison table in the README is generated from the two artefacts by
  `scripts/compare_naive_baseline.py`, and CI checks that it matches them.

## Amendment of 2026-11-05: completing the comparison

Written after both runs and their 23-question comparison were published, and committed
before any completion run. It adds runs; it changes no metric, no reading rule and no
setting of either arm. The 23-question block stays published as it is.

**Why.** Both arms stopped on their budget, and the questions run in id order. The
comparison therefore rests on 2 of the 13 unanswerable questions. Completing the
registered 39 involves no choice of questions, so it is the neutral way to get them
back.

**What runs.** Each arm runs the dev questions it did not complete, once; no completed
question runs again. The rows are copied unchanged from `dev.jsonl` by
`scripts/build_completion_subset.py` (CI: `--check`), in the order `random.Random(20261105)`
gives the sorted ids:

- Verity: `datasets/eval/dev-completion.jsonl`, sha256 `5a9a3332348c…`, 16 questions,
  5 answerable and 11 unanswerable.
- Naive: `datasets/eval/dev-completion-11.jsonl`, sha256 `8f6ddef6fb7c…`, 11 questions,
  2 answerable and 9 unanswerable. The same order, minus the 5 questions the naive arm
  completed.

If a budget stops a run again, the questions left out are then a random draw rather
than the end of the id order.

**Same configuration.** Models, judge, prompts, store, `VERITY_CONFIDENCE_THRESHOLD=0.0`,
and the support check at 0.5 and 0.0, all as registered above. The runs are made at a
commit whose `src/`, `prompts/`, `migrations/` and `datasets/corpus/` are identical to
`e644a98`; `scripts/run_naive_baseline.sh completion` refuses to start otherwise. The
false-refusal analysis (`docs/audit/2026-11-05-false-refusals.md`) changes no setting
before these runs.

**Analysis.**

- Each arm's rows are the union of its `e644a98` artefact and its completion artefact;
  a question appears once per arm.
- The comparison runs on the questions both arms completed across their two runs, with
  the metrics and the Wilson-overlap reading rule above, and names any question left
  out.
- It is published next to the 23-question block, labelled as the amended comparison.
- The runs start only once `scripts/compare_naive_baseline.py` reads two artefacts per
  arm. The result is published whichever way it comes out.

**Budget.** Estimated from the per-question cost (agent plus judge) of the `e644a98`
runs, whose pipeline code is the current one:

| Arm | Answerable | Unanswerable | Estimate |
|---|---|---|---|
| Verity | 5 × $0.0431 | 11 × $0.0114 | $0.34 |
| Naive | 2 × $0.0457 | 9 × $0.0339 | $0.40 |

- The Verity unanswerable mean rests on 2 questions. As a cross-check, the same 16
  questions cost $0.19 in the `6ec544d` run; the current code costs 1.56× that on the
  23 questions both runs share, which gives $0.29.
- The estimate of the first runs ($1.91) was made from older per-question costs, and
  the runs cost more.
- Caps coded in `scripts/run_naive_baseline.sh completion`: Verity $0.45, naive $0.52.
- Correction to "Budget and execution" above: the harness checks its cap after each
  question, not each call. An arm can therefore pass its cap by its dearest question:
  $0.087 (Verity) and $0.075 (naive) on the `e644a98` runs.
- Worst case: $0.45 + $0.52 + $0.16 = $1.13. Amount asked for: **$1.15**.
  `tests/unit/test_completion_subset.py` holds the caps to it.
