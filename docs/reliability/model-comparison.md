# Which model should Minerva run

A comparison of the tool-capable local models on the host, each with thinking off and on. This page fixes the
protocol **before any run**; results are added under "Results" with the model, digest, thinking state and reply budget
on every line. Changing anything under "Fixed in advance" after a run has started needs a dated entry saying why.
Nothing here changes the deployed model: that decision, the host and `HELM_EXTRA_VALUES` stay with the maintainer.

## Why

Every live result before 2026-11-19 was measured on one model, the deployed alias `minerva` (based on gemma4:26b,
25.2B, Q4_K_M, digest `149a0ff8a1ca`), with thinking off and the host's default reply limit of 1200 tokens. Nobody
had checked it was the right model. The only earlier comparison (2026-11-13, an older case set, 2 repeats) had
`qwen3.5:35b` at 66.2% against gemma's 48-51%, at 2-3 times the latency.

First thinking measurement (2026-11-19, `minerva`, digest `149a0ff8a1ca`, host default reply limit 1200, seed 7,
3 repeats, `--suite all --cases` the 11 cases where the baseline run called no tool): passes 1/33 -> 9/33, no-tool
runs 29 -> 19, false verified 0 -> 0, median turn 4.1 s -> 13.7 s, output tokens 5,976 -> 33,347, and 7 of 33 runs
cut off at the 1200-token limit (thinking spends the reply budget). So thinking helps on this subset, costs about
three times the latency, and needs a reply budget larger than the host default to be measured fairly.

## Fixed in advance

- **Models**: `minerva` (deployed service), `gemma4:26b` (control for the alias build and for the service),
  `qwen3.5:35b`, `qwen3.6:27b`, `qwen3.8:27b`, `granite4.2:30b`, `glm-4.7-flash` (system service). Thinking off, and
  on where the host reports the capability.
- **Reply budget**: `--num-predict 4096`, in both arms and for every model. The `minerva` thinking-off reference at
  4096 is **re-measured**, not taken from the 2026-11-19 baseline (which ran at the host default of 1200); the old
  baseline is quoted beside it as the effect of the budget alone.
- **Chat latency ceiling** (whole turn on this host): median <= 15 s and p95 <= 40 s, about twice the 2026-11-19
  baseline (7.0 s, 20.2 s). A configuration above either is reported and cannot be recommended as the default; it may
  be recommended as an opt-in thorough mode. The numbers are the maintainer's to change here before Stage 1 starts.
- **GPU**: the share of the loaded model that sits on the GPU is recorded for each run (`/api/ps`). Under 99%, or not
  recorded, the latency is not attributed to the model and the configuration is not eligible as the default.
- **Decision rule** (implemented in `agent/evaluation/matrix.py`, `eval_models.py --matrix`): no false verified result;
  within the ceiling and fully on the GPU; then pass rate, counted only when the gap is larger than the noise of both
  runs (`max(1, 2 x the larger std)` passes per repeat, as in `--compare`); ties go to the lower p95. Per dimension,
  no single score.
- **Stage 1 (screen)**: the frozen probe suite plus the held-out suite (44 cases), 3 repeats, seed 7,
  `--compute-timeout 10`, routing on, code at `main`. The cut keeps **every eligible configuration within noise of
  third place** (`matrix.screen`), not a fixed number. Configurations over the ceiling or with a false verified result
  are dropped first.
- **Stage 2 (full)**: what survived, plus the `minerva` thinking-off reference, on `--suite all`, 3 repeats.
- The model's digest is read from the host before and after every run; a changed digest stops the comparison.
- Runs are strictly sequential. Runs on a case set other than the one of the other rows are never ranked together.

## Where it runs

One host, one integrated GPU with unified memory (about 120 GB, 77 GB free when checked), two Ollama services that
share it. The deployed service keeps one model loaded (`OLLAMA_MAX_LOADED_MODELS=1`), so loading anything else there
would evict `minerva` for live users: `minerva` is run on the deployed service (it stays the loaded model), the other
models on the system service. The remaining risk to live users is GPU contention while a run goes on, and the
measured latencies include contention in both directions. Both services' versions and settings are recorded in the
reports. Each report shows what the host actually loaded for the model: window, size, share on the GPU.

## Cost and consequences

- Host time: up to 14 configurations in Stage 1 at roughly 30-90 minutes each; Stage 2 is 80 minutes for `minerva`
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

## Results

*(none yet: added as runs complete, each with model, digest, thinking, reply budget and report file)*

| Date | Report | Model (digest) | Thinking | Reply budget | Cases | Passed | Median / p95 s | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-19 | not committed (11-case subset) | minerva (`149a0ff8a1ca`) | off | host default 1200 | 11 no-tool cases x 3 | 1/33 | 4.1 / 21.4 (max) | reference for the thinking-on row |
| 2026-11-19 | not committed (11-case subset) | minerva (`149a0ff8a1ca`) | on | host default 1200 | 11 no-tool cases x 3 | 9/33 | 13.7 / 58.4 (max) | 7 of 33 cut off at 1200 tokens; false verified 0 |
