# Mutation testing

Measured mutation score of the modules that decide what Verity ships, against the fast
test lane. A surviving mutant is a change to the code that no test notices; every one
listed here is either killed by a test or explained as equivalent.

## Run

| | |
|---|---|
| Date | 2026-11-03 |
| Commit (code and tests measured) | `599242f` |
| Tool | mutmut 3.8.1, Python 3.12, macOS (APFS) |
| Command | `make mutation` (`rm -rf mutants && OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES HF_HUB_OFFLINE=1 uv run mutmut run && uv run mutmut results --all true`) |
| Tests run per mutant | fast lane (`-m "not integration and not slow"`), minus the three files below |
| Wall time | 150 s for 1050 mutants |

## Score

| Module | Killed | Timeout | Survived | Total | Detected |
|---|---:|---:|---:|---:|---:|
| `agent/support.py` | 60 | 0 | 0 | 60 | 100 % |
| `agent/synthesizer.py` | 106 | 0 | 4 | 110 | 96.4 % |
| `eval/headline.py` | 581 | 1 | 22 | 604 | 96.4 % |
| `eval/refusal.py` | 34 | 0 | 0 | 34 | 100 % |
| `eval/retrieval_metric.py` | 124 | 0 | 7 | 131 | 94.7 % |
| `llm/json_output.py` | 61 | 0 | 5 | 66 | 92.4 % |
| `retrieval/fusion.py` | 44 | 0 | 1 | 45 | 97.8 % |
| **Total** | **1010** | **1** | **39** | **1050** | **96.3 %** |

The timeout counts as detected, not as a pass: the mutant (`_refusal_counts`,
`tp / (tp + fp)` → `tp / (tp - fp)`) changes behaviour, and the suite ran past mutmut's
time limit on it instead of passing. The cause of the slowdown was not investigated.
No mutant was reported as "no tests", "segfault" or "suspicious".

The first clean run of this configuration, at `4163b0f`, left 80 survivors; 41 were
killed by the tests added in `599242f`. All 39 that remain are equivalent mutants (no input
distinguishes them), dead branches, or one platform artefact, listed below.

## Why these modules

They turn model output and retrieval results into what a user or a reader of the
scorecard sees:

- `agent/synthesizer.py`: parses the draft and keeps only verbatim citations.
- `agent/support.py`: the support checks that gate a draft.
- `llm/json_output.py`: recovery of JSON from model output, shared by synthesizer,
  scorer and judge.
- `eval/refusal.py`, `eval/retrieval_metric.py`, `eval/headline.py`: the numbers
  published in the scorecard.
- `retrieval/fusion.py`: RRF, which orders the evidence the model sees.

Left out: I/O adapters (API, stores, clients, CLI), whose behaviour depends on
services the fast lane stubs, and modules whose logic is mostly delegated to libraries
(embedder, reranker, docling parser).

## Configuration notes

- `process_isolation = "forkserver"` and `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES`:
  with plain fork, workers forked from the process that already ran the suite died on
  unmutated code (reported as segfault, exit -11 or -6). With both settings, 0 mutants
  crash.
- `tests/unit/test_request_log.py`, `tests/unit/test_tracer.py` and
  `tests/unit/test_chunker.py` are not run against mutants. They assert on
  `structlog.testing.capture_logs`; mutmut calls pytest several times in one process,
  and loggers cached on first use by an earlier call bypass the capture, so these files
  fail on unmutated code. None of them targets a mutated module.
- Not in CI: 150 s on a laptop for one platform, on top of the existing jobs, for a
  number that changes only when these modules or their tests change. `make mutation`
  reproduces it.

## Surviving mutants

### `llm/json_output.py` (5)

| Mutant | Change | Why it survives |
|---|---|---|
| `first_json_object` 16 | `end != -1` → `end != -2` | Equivalent: with `start >= 0`, `end == -1` already fails `end > start`. |
| `first_json_object` 17 | `end > start` → `end >= start` | Equivalent: one character cannot be both `{` and `}`. |
| `first_json_object` 35 | `array_end != -1` → `!= -2` | Same as 16 for arrays. |
| `first_json_object` 36 | `array_end > array_start` → `>=` | Same as 17 for arrays. |
| `parse_json_object` 6 | first dict of a list → `None` | Dead branch: a list holding a dict always contains `{` and `}`, so `first_json_object` returns the brace slice and the decoded value is never a list holding a dict. See open item 1. |

### `agent/synthesizer.py` (4)

| Mutant | Change | Why it survives |
|---|---|---|
| `__init__` 2 | `"synthesize"` → `"SYNTHESIZE"` | Not killed on a case-insensitive file system (APFS resolves `prompts/SYNTHESIZE.prompt` to the same file); on a case-sensitive one (Linux CI) `load_prompt` raises `FileNotFoundError` and every synthesizer test kills it. Asserting the private `_prompt.name` would kill it here too; left out to keep tests on the public surface. |
| `synthesize` 3 | `with_scores=False` → `None` | Equivalent: `render_evidence` only tests truthiness. |
| `_parse_synth_output` 17 | `get("citations", [])` → `get("citations", None)` | Equivalent: `or []` follows. |
| `_parse_synth_output` 19 | `get("citations", [])` → `get("citations")` | Same as 17. |

### `retrieval/fusion.py` (1)

| Mutant | Change | Why it survives |
|---|---|---|
| `fuse` 1 | `k <= 0` → `k < 0` | Equivalent: with `k = 0` the final `[:k]` slice is empty, the same `[]`. |

### `eval/retrieval_metric.py` (7)

| Mutant | Change | Why it survives |
|---|---|---|
| `recall_at_k` 6 | `k <= 0` → `k < 0` | Equivalent: `retrieved_ids[:0]` counts no hit, recall 0.0. |
| `ndcg_at_k` 6 | `k <= 0 or not ids` → `and` | Equivalent: for `k <= 0`, `min(k, |gold|) <= 0`, so the ideal DCG is 0 and the final guard returns 0.0; for `k > 0` with no ids, DCG is 0 and `0 / idcg` is 0.0. |
| `ndcg_at_k` 7 | `k <= 0` → `k < 0` | Same: `k = 0` reaches the final guard, 0.0. |
| `ndcg_at_k` 36 | `idcg > 0` → `(idcg > 0) or True` | Unreachable: with a non-empty gold set (checked first) and `k >= 1`, the ideal DCG is at least 1. |
| `ndcg_at_k` 38 | `idcg > 0` → `idcg >= 0` | Same as 36. |
| `ndcg_at_k` 40 | `else 0.0` → `else 1.0` | Same as 36. |
| `BinaryRetrievalMetric.score` 5 | `_hit_ids(hits, k)` → `_hit_ids(hits, None)` | Equivalent: each metric slices `[:k]` again. |

### `eval/headline.py` (22)

| Mutant | Change | Why it survives |
|---|---|---|
| `_refusal_counts` 42–45 | keys `tp` / `fp` renamed | Dead output: callers read only `fn`, `precision` and `recall`. |
| `build_headline_dict` 5, 21 | provider stamp → always `None` | Equivalent: the provenance summaries count models only; the provider is discarded. |
| `build_headline_dict` 14, 17 | `getattr(result, "judge_stamps", [])` default changed or dropped | Unreachable default: `HarnessResult` always has `judge_stamps`. |
| `write_headline_scorecard` 10, 50 | `encoding="utf-8"` → `"UTF-8"` (read, write) | Equivalent: Python codec names are case-insensitive. |
| `write_headline_scorecard` 44, 46 | write with `encoding=None` / omitted | Equivalent for any ASCII-compatible locale: `json.dump` escapes non-ASCII, so the bytes are ASCII. |
| `write_headline_scorecard` 8 | read with `encoding=None` | Environment-dependent: the locale encoding is UTF-8 here (`locale.getpreferredencoding()`; POSIX coerces the C locale to UTF-8), so only a non-UTF-8 locale reading a non-ASCII file written by something else would tell it apart. Not testable in-process. |
| `write_headline_scorecard` 11 | unreadable file → `""` instead of `None` | Equivalent: both are "not a dict", which takes the same path. |
| `_existing_is_stub` 3, 5 | `get("provenance", {})` default → `None` or dropped | Equivalent: a missing provenance is "not a stub" either way (the `isinstance` check returns `False`; `{}` yields empty names). |
| `_existing_is_stub` 14, 16, 19, 24, 26, 29 | `get("agent_model" / "judge_model", "")` default → `None`, dropped or `"XXXX"` | Equivalent: `"none"` and `"xxxx"` contain no `"stub"`, like `""`. |

## Open items found by this run

1. `parse_json_object`'s docstring says an array's first dict is taken as the object.
   That holds only for a one-dict array: `[{"a": 1}, {"b": 2}]` yields the slice
   `{"a": 1}, {"b": 2}`, which is invalid, so the result is `None`. This fails closed
   (the scorer and synthesizer refuse). The list branch it describes is unreachable.
2. The `tp` / `fp` keys of `_refusal_counts` are never read.
3. `build_headline_dict` reads `getattr(result, "judge_stamps", [])` although
   `HarnessResult.judge_stamps` is a field with a default; the fallback is dead and is
   the only reason survivors 14 and 17 exist.
