# Which model should Minerva run

A comparison of the tool-capable local models on the host, each with thinking off and on. This page fixes the
protocol **before any run**; results are added under "Results" with the model, digest, thinking state and reply budget
on every line. Changing anything under "Fixed in advance" after a run has started needs a dated entry saying why.
Nothing here changes the deployed model: that decision, the host and `HELM_EXTRA_VALUES` stay with the maintainer.

## Why

Every live result before 2026-11-19 was measured on one model: Gemma 4 26B (`gemma4:26b`, a mixture-of-experts model,
25.2B, Q4_K_M, the deployed build, digest `149a0ff8a1ca`). The deployed host serves it under the alias `minerva`, which is
also the product's name and says nothing about the weights, so this page and every report name the model by what it
is and give the alias only as "served as", with thinking off and the host's default reply limit of 1200 tokens. Nobody
had checked it was the right model. The only earlier comparison (2026-11-13, an older case set, 2 repeats) had
`qwen3.5:35b` at 66.2% against gemma's 48-51%, at 2-3 times the latency.

First thinking measurement (2026-11-19, `gemma4:26b` deployed build, digest `149a0ff8a1ca`, host default reply limit 1200, seed 7,
3 repeats, `--suite all --cases` the 11 cases where the baseline run called no tool): passes 1/33 -> 9/33, no-tool
runs 29 -> 19, false verified 0 -> 0, median turn 4.1 s -> 13.7 s, output tokens 5,976 -> 33,347, and 7 of 33 runs
cut off at the 1200-token limit (thinking spends the reply budget). So thinking helps on this subset, costs about
three times the latency, and needs a reply budget larger than the host default to be measured fairly.

## Fixed in advance

- **Models**: `gemma4:26b`, deployed build served as `minerva` (deployed service), `gemma4:26b` (control, system service),
  `qwen3.5:35b`, `qwen3.6:27b`, `qwen3.8:27b`, `granite4.2:30b`, `glm-4.7-flash` (system service). Thinking off, and
  on where the host reports the capability.
  **Added 2026-11-19, before any of these ran** (the first four Stage 1 configurations had already run): the Mistral
  family on the host, `mistral-small3.2:24b` (thinking off only: the host reports no thinking mode), `magistral:24b`
  (off and on) and `ministral-3:14b` (off only). Adding models widens the screen, so more configurations can look
  good by chance; the within-noise rule and the Stage 2 confirmation on the full suite are what guard against that.
- **Reply budget**: `--num-predict 4096`, in both arms and for every model. The `gemma4:26b` (deployed build) thinking-off reference at
  4096 is **re-measured**, not taken from the 2026-11-19 baseline (which ran at the host default of 1200); the old
  baseline is quoted beside it as the effect of the budget alone.
- **Speed is reported, never the deciding factor** (amended 2026-11-19 on the maintainer's instruction, after the first
  Stage 1 results; the original text made a latency ceiling a condition for being the default): this is a bench of a
  math tool, a chat may be slow when it is accurate. Every result is given twice, **at any speed** (the main result) and
  **within the chat ceiling** (median <= 15 s and p95 <= 40 s per turn, about twice the 2026-11-19 baseline of 7.0 s and
  20.2 s), so a slow configuration is never hidden and the cost of choosing it is stated. The ceiling no longer drops a
  configuration from the screen. `qwen3.6:27b` (median 22.7 s, p95 57.3 s with thinking off) was the first affected.
- **GPU**: the share of the loaded model that sits on the GPU is recorded for each run (`/api/ps`). Under 99%, or not
  recorded, the latency is not attributed to the model and the configuration is not eligible as the default.
- **Decision rule** (implemented in `agent/evaluation/matrix.py`, `eval_models.py --matrix`): no false verified result
  and fully on the GPU; then pass rate, counted only when the gap is larger than the noise of both
  runs (`max(1, 2 x the larger std)` passes per repeat, as in `--compare`); ties go to the lower p95. Per dimension,
  no single score.
- **Stage 1 (screen)**: the frozen probe suite (38 cases, 44 turns; `--suite probe`), 3 repeats, seed 7,
  `--compute-timeout 10`, routing on, code at `main`. The cut keeps **every eligible configuration within noise of
  third place** (`matrix.screen`), not a fixed number. Configurations with a false verified result or not fully on
  the GPU are dropped first; slow ones stay.
- **Stage 2 (full)**: what survived, plus the `gemma4:26b` (deployed build) thinking-off reference, on `--suite all`, 3 repeats, and the
  held-out suite (6 cases) for the survivors only. (2026-11-19, before any run: the held-out suite was moved here from
  Stage 1 because one report holds one case set and `--matrix` refuses to rank across case sets.)
- The model's digest is read from the host before and after every run; a changed digest stops the comparison.
- Runs are strictly sequential. Runs on a case set other than the one of the other rows are never ranked together.

## Where it runs

One host, one integrated GPU with unified memory (about 120 GB, 77 GB free when checked), two Ollama services that
share it. The deployed service keeps one model loaded (`OLLAMA_MAX_LOADED_MODELS=1`), so loading anything else there
would evict the deployed model for live users: the deployed build is run on the deployed service (it stays the loaded model), the other
models on the system service. The remaining risk to live users is GPU contention while a run goes on, and the
measured latencies include contention in both directions. Both services' versions and settings are recorded in the
reports. Each report shows what the host actually loaded for the model: window, size, share on the GPU.

## Cost and consequences

- Host time: up to 14 configurations in Stage 1 at roughly 30-90 minutes each; Stage 2 is 80 minutes for the deployed build
  and likely 2-4 times that for larger or thinking configurations. More than a day in total, during which chat replies
  for live users may be slower.
- **A different model, thinking state or reply budget invalidates the homepage approvals.** The four approved example
  cards and the three Explore entries each need five clean live runs on the new configuration. The owed Explore runs
  wait for this decision.

## Known limits

- The prompt, repairs and routing were tuned on traces of the gemma build, so the comparison favours it.
- The probe suite was written by the author of the fixes; the held-out suite has 6 cases. Stage 1 is a screen, not a
  ranking.
- Local Q4 builds only, no cloud model. Latency is this host's and includes contention.

## Results (2026-11-19 and 2026-11-20)

Every row names the model, its digest and whether thinking was on. All runs: seed 7, 3 repeats, `--num-predict 4096`,
`--compute-timeout 10`, routing on, every model fully on the GPU (100% of its loaded size, as the host reported).
Reports are committed in [`docs/benchmarks/model-comparison-2026-11-20/`](../benchmarks/model-comparison-2026-11-20/)
(raw JSON of every run, and the tables produced from them by `eval_models.py --matrix`). Latencies were measured on a
host shared with other work on this machine and are approximate; accuracy does not depend on them.

### The deployed model, by what it is

`gemma4:26b` (a 25.2B mixture-of-experts model, Q4_K_M, digest `149a0ff8a1ca`, **served by the host as `minerva`**). Plain
`gemma4:26b` on the other service (digest `001e5dafc3c7`) gave the same results run for run (101/114 off, 100/114 on).
The host reports the parent of `qwen3.5:35b` as `qwen35-validation-20261004:35b-a3b-q4`, an internal name; the model is
Qwen 3.5 35B-A3B (a mixture of experts, 34.7B, Q4_K_M).

### Stage 1: screen on the probe suite (38 cases, 114 runs per configuration)

| Model | Digest | Type | Thinking off | Thinking on | Median s (off / on) |
| --- | --- | --- | --- | --- | --- |
| `gemma4:26b` (deployed build, served as `minerva`) | `149a0ff8a1ca` | mixture of experts | 101 (88.6%) | 100 (87.7%) | 4.4 / 5.6 |
| `gemma4:26b` (system service) | `001e5dafc3c7` | mixture of experts | 101 (88.6%) | 100 (87.7%) | 3.2 / 5.5 |
| `qwen3.5:35b` | `72aac7c28be8` | mixture of experts | 97 (85.1%) | **106 (93.0%)** | 6.4 / 7.9 |
| `qwen3.6:27b` | `bcbdbd4bcf86` | dense | 97 (85.1%) | 104 (91.2%) | 22.7 / 28.0 |
| `qwen3.8:27b` | `aaee06c39dcf` | dense | 97 (85.1%) | 104 (91.2%) | 18.1 / 24.4 |
| `glm-4.7-flash` | `4475827791a2` | mixture of experts | 104 (91.2%) | 103 (90.4%) | 5.3 / 8.8 |
| `granite4.2:30b` | `be22829aa92a` | dense | 98 (86.0%) | 90 (78.9%) | 17.9 / 61.6 |
| `mistral-small3.2:24b` | `5a408ab55df5` | dense | 69 (60.5%) | not offered | 16.9 |

Not completed, by the maintainer's decision to narrow the comparison: `magistral:24b` (stopped after 32 of its 114 runs,
thinking off) and `ministral-3:14b` (never run). They are not measured; nothing here ranks them.

What the screen showed: thinking helps the three qwens (+7 to +9 passes) and tool selection reaches 99-100% with it;
it does nothing for the gemma and glm builds (a difference of 1) and hurts granite (-8 passes, 61.6 s median). The dense
27-30B models are three to five times slower than the mixture-of-experts ones at similar accuracy.

### Stage 2: the full suite (core and reliability cases, 510 runs per configuration) and the held-out suite

| Model | Digest | Thinking | Passed | Per repeat | Tool selection | Valid arguments | Completion | False verified | Median / p95 s | Held-out (18 runs) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen3.5:35b` | `72aac7c28be8` | on | **389 (76.3%)** | 128, 130, 131 | **95.3%** | 83.5% | 92.4% | **2** (see below) | 12.4 / 43.8 | 16 |
| `gemma4:26b` (deployed build, served as `minerva`), the reference | `149a0ff8a1ca` | off | 372 (72.9%) | 126, 125, 121 | 87.3% | 88.0% | 93.7% | 0 | 6.1 / 18.9 | 17 |
| `glm-4.7-flash` | `4475827791a2` | off | 316 (62.0%) | 108, 104, 104 | 86.1% | 82.5% | 83.3% | 0 | 10.2 / 51.0 | 17 |

For the reference at the earlier host default reply limit of 1200 tokens (the 2026-11-19 baseline): 363/510 (71.2%).
The difference to 372 is inside the noise, so the larger reply limit alone changed nothing measurable.

### What the comparison found

1. **`qwen3.5:35b` with thinking on is the most accurate configuration measured, beyond the run-to-run noise**: 5.7 passes
   per repeat ahead of the reference against a noise limit of about 4.3. It is also the only one that meets the
   tool-selection target (95.3%) and never failed to call a tool. It is about twice as slow, has weaker argument
   validity (83.5% against 88.0%) and more model-honesty failures (42 failed runs against 24).
2. **The screen over-predicted**: `glm-4.7-flash` scored 91.2% on the probe suite and 62.0% on the full suite (it loops:
   68 failed runs of that class against 23 for the reference). The other two lost 16-17 points moving from the probe
   suite to the full suite. A small suite written by the people fixing what it measures is a
   screen, not a ranking.
3. **No configuration meets the pass-rate (85%), valid-arguments (99%) or completion (95%) target.** None is close on
   valid arguments.
4. **Thinking is model-specific.** It helps qwen, is neutral for gemma and glm, and hurts granite; it cannot be assumed.

### The false-verified results, and what they turned out to be

The comparison produced three false-verified counts, and none was a model failing; two were defects in our code, found
only because a different model wrote different prose:

- `mistral-small3.2:24b`, probe suite, 2: its reply was correct (the identity needs `AB = BA`). The evaluation harness
  gave the claim checker the declared matrices only for multi-turn cases, so a false identity was marked "checked".
  The chat route was not affected. **A-008**, PR 117; re-run on the fixed harness: 0.
- `qwen3.5:35b`, full suite, 2: after a conversation declared `A` and `B` as matrices, the model called `check_identity`
  on `(A+B)^2 = A^2 + 2AB + B^2` and **our tool answered "proved equal"** (it samples commuting numbers), and the loop
  marked the formula as from the tool. `check_identity` was missing from the list of tools that refuse a declared matrix.
  A probe of the other tools found `rewrite_part`, `sums` and the Lagrangian tools with the same hole. **A-009**, PRs 121
  and 123. Re-run of that case on the fixed code (`main` at `ab50f75`), 5 repeats: 5 passed, false verified 0.
- The same comparison found a false "contradicted" (the correct curvature value at `r = 2M`, **A-010**, PR 124) and a
  probe search that a loaded machine cut short without saying so (**V-017**, PR 125, visibility only).

Under the rule fixed in advance a false-verified run is not eligible, so `qwen3.5:35b` is **not eligible as measured**.
The defect is fixed and the case re-run is clean, but its full suite has not been re-run on the fixed code. Until it is,
the honest statement is: most accurate as measured, with a disqualifying result that has since been fixed and
confirmed on the one case.

### Recommendation

- **Keep `qwen3.5:35b` with thinking on as the chat model** (the deployed configuration was switched to it on
  2026-11-19 at the maintainer's request, before this comparison finished; the evidence supports the choice, with the
  conditions below). Do not move to `glm-4.7-flash`, granite or Mistral.
- **Confirm it as deployed.** The deployed instance's `qwen3.5:35b` has digest `c2ee66a3c0d6`; the copy measured here has
  `72aac7c28be8`. The tag is the same and the weights may be, but this is unverified. A full-suite run on the deployed
  instance, on the fixed code, would settle both the digest question and the eligibility of the false-verified rule
  (about 2.75 hours; it loads the live service).
- **The cost of being slower** is a median turn near 12 s (shared host) against 6 s; the maintainer judged accuracy
  to outweigh speed for a mathematics tool.
- **Re-record or re-approve what is tied to the model.** The landing-page sessions are recorded on this configuration and
  guarded by a test. The homepage example cards and the Explore entries were approved on the earlier model and need five
  clean live runs each on this one.

### What was changed along the way (all dated 2026-11-19 or 2026-11-20)

The protocol above was adjusted four times, each on the maintainer's instruction and each before the results it affects:
Stage 1 screens on the probe suite alone (one report holds one case set); the Mistral models were added; the chat latency
ceiling stopped deciding (speed is reported apart; accuracy decides); and the Mistral queue was stopped after
`mistral-small3.2:24b` (the narrowing to two winners plus the reference). Which two: `qwen3.5:35b` with thinking on, the
most accurate on the screen, and `glm-4.7-flash` with thinking off, tied with the 27B qwens on the screen at a quarter of
their time. `glm-4.7-flash` did not hold up; the other qwens, granite and the gemma build were not carried to Stage 2
beyond the reference, so Stage 2 does not say whether `qwen3.6:27b` or `qwen3.8:27b` would match `qwen3.5:35b` on the full
suite (they tied it on the screen and are 2-3 times slower).

### Earlier measurements

| Date | Report | Model (digest) | Thinking | Reply budget | Cases | Passed | Median / p95 s | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-19 | `thinking-subset-...-thinking-on.json` (11-case subset) | `gemma4:26b` deployed build (`149a0ff8a1ca`) | on | host default 1200 | 11 no-tool cases x 3 | 9/33 | 13.7 / 58.4 (max) | against 1/33 with thinking off in the 2026-11-19 baseline; 7 of 33 cut off at 1200 tokens; false verified 0 |
