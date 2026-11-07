# Pre-registration: Verity against a naive RAG baseline on all 95 questions

Committed before any run it describes. The arms, metrics, tests and decision rule
below are fixed by this file; the result is published as it comes out, favourable or
not. It replaces the dev-only comparison of `naive-rag-baseline.md`: 23 paired
questions, of which 2 unanswerable, stopped by its spending cap.

## Question

Same questions, same models, same retrieval, same synthesis prompt: do Verity's checks
refuse unanswerable questions that the synthesizer's own draft answers, and what do
they cost on answerable ones?

## Data and models

- Datasets: `datasets/eval/dev.jsonl` (39 questions, 26 answerable) and
  `datasets/eval/test.jsonl` (56, 36 answerable), each run as its own artefact; the
  analysis pairs the arms over the 95 questions: 62 answerable, 33 unanswerable.
- Agent model `claude-sonnet-4-6` in both arms. Judge `claude-opus-5-5` with
  `prompts/judge_faithfulness.prompt` v0.2.0, the judge the code ships, in both arms.
- Store: in memory (`--store memory`). Prompts as in the repository at the commit of
  the runs.

## Arms

Both arms run at one commit, one after the other, once each, on each split.

- **Verity**: the shipped defaults at that commit: confidence threshold 0.0 (it
  refuses nothing), cited-relevance check off, citation coverage floor 0.5, injection
  markers on. A draft is refused when it has no verified citation, when its citations
  cover less than half of it, when it cites a document with injection markers, or
  when the scorer refuses.
- **Naive RAG** (`--naive-baseline`): same decomposition, retrieval and synthesis; the
  draft ships as it is, with no check and no scorer call.

## Metrics

**Self-abstention.** The naive arm records no refusal, so every naive draft counts as
answered and "Unanswerable answered" would favour Verity by construction. In both arms,
an answer whose text matches one of the patterns of `naive-rag-baseline.md` (metric 4)
is counted as a refusal in the three primary metrics.

Primary, each an exact two-sided McNemar test on the paired questions
(`verity.eval.paired.mcnemar_exact`), Holm-corrected over the three at family-wise
α = 0.05:

1. **Unanswerable answered** (33 pairs): the question got an answer that is not a
   self-abstention. Lower is better.
2. **Answerable answered** (62 pairs): same definition. Higher is better; it is what
   the checks cost.
3. **Invented answer** (95 pairs): the question got an answer that is not a
   self-abstention and that has no claim or a claim the judge finds unsupported
   (`verity.eval.answer_quality.is_invented`). A refusal invents nothing. Lower is
   better.

Each arm's proportions are reported with Wilson 95 % intervals, and metric 3 also as a
rate over the answers it shipped.

Secondary, descriptive, no test: the three metrics without the self-abstention rule;
hallucination rate on answerable questions; correctness against `reference_answer`
(`scripts/judge_correctness.py`) on the answerable questions each arm answered; refusal
recall per tag; cost per query, with and without the judge.

## Power

`uv run python scripts/power_two_arm.py`, at the Holm first-step level 0.05 / 3 (the
strictest any test faces); 7 discordant pairs all one way are the minimum to reject.
The rates are those of a question going to one arm only.

| Test | n | P(naive only) | P(Verity only) | Expected discordant | Power |
|---|---|---|---|---|---|
| Unanswerable answered | 33 | 0.30 | 0.00 | 9.9 | 0.906 |
| Unanswerable answered | 33 | 0.50 | 0.00 | 16.5 | 1.000 |
| Answerable answered | 62 | 0.05 | 0.00 | 3.1 | 0.035 |
| Answerable answered | 62 | 0.10 | 0.00 | 6.2 | 0.427 |
| Answerable answered | 62 | 0.15 | 0.00 | 9.3 | 0.841 |
| Invented answer | 95 | 0.05 | 0.00 | 4.8 | 0.198 |
| Invented answer | 95 | 0.10 | 0.01 | 10.4 | 0.552 |

Anchors: on the test run of record Verity refused 5 of 36 answerable questions with
the cited-relevance check on (`scorecards/gate-ablation.json`); with it off the ablation
bounds that at 1 to 5. A cost on answerable questions of 5 points or less is therefore
not detectable here, and a difference in invented answers below about 10 points is not
either. A non-significant result on metrics 2 and 3 reads as "not measurable at this
n", never as "no difference".

## Decision rule

Written before the runs; the published text uses these sentences.

- Metric 1 significant, Verity lower: "Verity's checks refuse unanswerable questions
  that the naive draft answers."
- Metric 1 not significant: "On 33 unanswerable questions the checks show no effect
  beyond the synthesizer's own abstentions." Design implication: the refusal checks are
  not shown to earn their cost, and the comparison that decides it is the full-corpus
  arm below on a larger question set.
- Metric 1 significant, Verity higher: published as is; the checks are then a defect.
- Metric 2 significant, Verity lower: "the checks cost answerable answers", with both
  discordant counts; the design question becomes the trade between metrics 1 and 2.
  Not significant: "no cost measurable at this n" with the power above.
- Metric 3: the same two readings, with the power above.

No re-run, no change of metric after a run, no pooling with the e644a98 runs.

## Budget and execution

- Measured cost per question on the runs of record: Verity agent ≈ $0.022, judge
  ≈ $0.0185 per judged answer (e644a98 dev); naive agent ≈ $0.017, judge ≈ $0.027
  (e644a98 dev naive). Estimate: Verity dev $1.36, test $1.95; naive dev $1.72,
  test $2.46; total ≈ $7.5.
- Caps in `scripts/run_two_arm.sh`: Verity dev $1.60, test $2.40; naive dev $2.00,
  test $2.90; total $8.90. The harness checks its cap after each call, so each run can
  pass it by one call (≈ $0.03). Correctness grading of both arms: cap $0.70. With the
  $0.19 already spent in this mission, the worst case is $9.91, under the owner's
  $12.00 ceiling.
- A run that hits its cap or has errored questions is published incomplete, with the
  questions it did not complete named; the tests then run on the questions both arms
  completed, and say so.
- Artefacts: `datasets/eval/runs/live/eval_<sha>_{dev,test}.json` and
  `eval_<sha>_{dev,test}-naive-baseline.json`, pinned in `SHA256SUMS`, never edited.

## The full-corpus arm: specified, not run

The corpus is 5,489 words and fits in one context, so "every chunk in the prompt, no
retrieval" is the obvious baseline. Its cost was measured before this file was
committed, with Anthropic's `count_tokens` endpoint, which is not billed, on the 73
chunks of the 21 documents (Markdown sources of the PDF and DOCX files), datamarked as
the synthesizer and judge render them: **28,106 input tokens per synthesis call and
32,538 per judge call**. At $3 and $4 per million input tokens, that is about $0.087
per question for the agent and $0.134 per judged answer: about $20 for 95 questions,
over the $12 ceiling of this mission on its own.

- Scope: this pre-registration covers two arms.
- Known limit: the full-corpus question stays open; nothing here says Verity beats it.
- Next step: put the evidence before the question in `prompts/synthesize.prompt` so
  the corpus becomes a cacheable prefix (the judge prompt already has it first), add
  prompt caching to the Anthropic client, and re-measure; or approve about $20 for the
  arm as specified.
