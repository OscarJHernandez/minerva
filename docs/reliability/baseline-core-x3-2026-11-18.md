# Core-suite baseline, three repeats (2026-11-18)

The reference point for the next round of reliability work. Reports: `docs/benchmarks/core-live-x3-baseline.json`
and `.md` (the report opens with the model under test). Traces of every run were kept locally
(`eval-traces/`, git-ignored: they hold prompts and replies).

## What was measured

| | |
| --- | --- |
| Model | `minerva` on the operator's Ollama host: based on `gemma4:26b`, 25.2B, Q4_K_M, digest `149a0ff8a1ca`, thinking-capable, thinking **off** |
| Host defaults | `num_predict 1200`, `num_ctx 16384` in the Modelfile; Minerva sent `num_ctx 32768` (loaded: 32768), temperature 0.2, seed 7 + repeat |
| Suite | core, 161 cases (148 scored, 13 informational), 3 repeats, `--compute-timeout 10` |
| Code | `main` at `064ec6e` (PR 94); routing on |

The "same weights through the run" line rests on a probe I read by hand from the host while the run was in
progress and on the identical reading after it (digest and modified time equal); the harness did not record the
model at the start of this run because the recording was added afterwards.

## Result

| Metric | Value |
| --- | --- |
| Pass | 267/444 runs (60.1%); per repeat 87, 90, 90 of 148 |
| Tool selection | 82.7% |
| Valid arguments | 83.7% |
| Completion | 91.7% |
| False verified | 0 |

**Noise.** The three repeats differ by at most 3 cases (stdev 1.4 cases, about one point). A change smaller than
that is not a change. Of 148 cases, 82 passed all three times, 48 failed all three times and 18 flipped.

## Where the failures are

177 failed runs by class: tool selection 61, model honesty 36, arguments 34, unproductive loop 33, parsing 6,
resource limit 4, missing assumptions 3. In 52 of them the model called no tool at all and in 41 at least one call
had invalid arguments. Only 2 runs ended on the output limit: the host's `num_predict 1200` is not a cause of
failure with thinking off, which is why it matters when thinking is switched on.

| Category | Pass | Tool selection |
| --- | --- | --- |
| multistep | 90.0% | 100% |
| retrieval | 81.5% | 81.5% |
| symbolic | 80.0% | 90.0% |
| followup | 72.2% | 100% |
| ambiguous | 64.4% | 93.3% |
| calculus | 63.4% | 93.5% |
| tensor_matrix | 62.5% | 70.8% |
| physics | 53.3% | 86.7% |
| **false_claim** | **35.7%** | **42.9%** |
| **unsupported** | **23.3%** | **50.0%** |

- **False claims ("is it true that A = B?") and unsupported requests are the two weak categories**, and in both the
  model mostly answers in prose without a tool. No tool decides "is this identity true?", so for false claims the
  model has nothing to call; this is the case for the planned `check_identity` tool.
- **`statistical_mechanics` passed 0 of 27 runs with 59 of 72 calls invalid**; `check_units` selection is 1 of 6;
  `vector_calculus` 13 of 120 calls invalid; `distribution` 11 of 24. The invalid-argument text names no field of
  a union-typed argument (`tools.py` `_shape_of`), which is the planned fix.
- **The wording checks.** 18 cases (44 runs) carry the note "reply does not say what it should". Two of them
  (`ambiguous-simplify`, `ask-when-ambiguous`, 6 runs) fail on that note alone: every reply was "Please provide
  the expression ..." and the check wanted a literal `?`. That check was too strict and now also accepts a clear
  request (case-set digest `d2d3164e5b44` -> `5ecf22d15a7d`; a guessed answer still fails). The other 16 cases fail
  the wording check together with a wrong or missing tool, so they are the model's failures, not the check's.
  With the two cases corrected the same runs would be 273/444 (61.5%): the check was not the main problem.

## What follows

Planned from this, in order: the `check_identity` tool for false-claim questions; argument-shape hints for union
arguments (`statistical_mechanics`); then, from thinking traces of the no-tool answers (PR 98 stores them), whether
routes are justified. These numbers are the "before": a change is reported against them with its noise.
