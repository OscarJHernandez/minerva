# Experiment log

Every live run against a model is one row here, written when it finishes, whatever the outcome (a run that changed
nothing and was reverted is as useful as one that worked). A row says **which model with which configuration**, which
code, which cases, how often, what it measured and what was decided. Reports live in `docs/benchmarks/`; this page is
the index and the decisions. Hand-written interpretation stays in `progress-2026-11-18.md` and
`baseline-core-x3-2026-11-18.md`; `docs/model-evaluation.md` is never overwritten.

## Rules for a row

- All live runs go to the operator's `metacon` Ollama host (the instance that serves the deployed `minerva` model).
  The address is never written down. A cloud model is used only with the maintainer's word.
- A row names the model as the host reports it (base model, size, quantisation, digest), thinking requested or not,
  the host's own defaults (`num_predict`), the window loaded, seed and repeats. Reports written since PR 96 carry this
  in their "Model under test" table; older rows say what is known and from where.
- Before and after share model digest, seed, repeats and `--compute-timeout`; if not, the row says so.
- Noise first: three repeats of the core suite differed by at most 3 cases of 148 (about one point). A difference
  inside that is "no effect".
- A change that did not generalise or did not beat the noise is reverted and its row says so.

## The model measured on 2026-11-18

`minerva` on metacon: based on `gemma4:26b`, 25.2B parameters, Q4_K_M, digest `149a0ff8a1ca`, modified
2026-11-15T16:30:28Z, capabilities completion, vision, tools, thinking. Host defaults: `num_predict 1200`,
`num_ctx 16384` (Minerva sends `num_ctx 32768`; 32768 loaded), `temperature 0.2`, `top_k 64`, `top_p 0.9`. Thinking
**off** in every row unless stated. Ollama 0.40.3. Rows before 2026-11-18 evening did not record the digest in the
report; the host reported this same digest and modified time when first read that afternoon and again after the
long runs, and nothing indicates it changed.

## Rows

| # | Run | Code | Cases x repeats | Result | Decision |
| --- | --- | --- | --- | --- | --- |
| 1 | `core-live-routing-off` / `-on` | `e5f9a50` | core, 161 (148 scored) x 1, seed 7 | 85/148 (57.4%) without the routes, 87/148 (58.8%) with; tool selection 81.1% / 82.4%; false verified 0 | Routes help their own shapes, not the broad suite (+1.4 points, inside the noise of a single run). Documented as such. |
| 2 | `core-live-subset-after-repairs` | `e5f9a50`+ | 29 affected cases x 1 | invalid calls 20/66 -> 10/61 (telescoping 10 -> 1); passes 9 -> 10 of 29 | Fewer invalid calls did not make cases pass. Kept (cheaper turns); not claimed as a pass-rate gain. |
| 3 | `reliability-live-*`, `heldout-live-*` (before/after routes) | see reports | reliability and held-out suites x 3 | see `progress-2026-11-18.md` | Held-out is not independent evidence (same author); recorded as the weaker figure. |
| 4 | `unsupported-prompt-before` / `-after` | `5253e5a` (dirty) | the 10 "unsupported" cases x 3, seed 7 | 5/30 -> 4/30; first tool 15/30 both; false verified 0 | A core-prompt rule ("call the tool even if you expect a refusal") **reverted**: same prose answers in all repeats. |
| 5 | `core-live-x3-baseline` | `064ec6e` | core, 148 scored x 3, seed 7 | 267/444 (60.1%); repeats 87/90/90; tool selection 82.7%; valid arguments 83.7%; false verified 0; 82 always pass, 48 always fail, 18 flip | The reference point and the noise (about one point). Wording checks classified: 2 cases (6 runs) failed only on an over-strict check (fixed, PR 99). |
| 6 | `statmech-shape-hints` | after PR 100's change | 9 `statmech-*` cases x 3, seed 7 | 0/27 -> 6/27; invalid calls 59/72 -> 36/73; repeated-call loops 9 -> 0; false verified 0 | Kept (PR 100). 7 of 9 cases still fail; before-numbers come from row 5 (a different run). |
| 7 | `false-claim-check-identity` | `b0912da` | 16 `false_claim` cases (14 old + 2 new) x 3, seed 7 | 15/42 (35.7%) -> 31/48 (64.6%); on the 14 old cases 25/42; no tool called 19 -> 4; `check_identity` in 27 runs; false verified 0; median turn 6.1 s -> 7.0 s | Kept. One regression found: `false-pythagoras` 2/3 -> 0/3. The model called `check_identity` and answered correctly; the case's `also_ok` list did not accept it (a stale check, not a model fault), see row 8. |
| 8 | `false-pythagoras-recheck` | `b0912da` + case fix | `false-pythagoras` x 3, seed 7 | 0/3 -> 3/3 | Case check corrected (it was an over-strict tool list; the reply was right). |
| 9 | informal probe, thinking on vs off | `main` | one question, not a benchmark | thinking returned 2,938 characters separately from a 176-character reply; 1,013 completion tokens against 66 | With the host's `num_predict 1200` shared by thinking and answer, harder prompts will hit the length limit. Thinking stays off by default until measured on the suite. |
| 10 | `statmech-deterministic-fixes-a` | PR 102, steps 1-3 | 9 `statmech-*` cases x 3, seed 7 | 6/27 -> 8/27; invalid calls 36/73 -> 22/77; runs ending on repeated calls 0 -> 4; calls 73 -> 77; median turn 17.7 s -> 16.6 s | Kept (unwrap a system nested under its kind, worked partials, chain syntax). `statmech-stirling-is-an-approximation` 0 -> 2 of 3. |
| 11 | `statmech-deterministic-fixes-b` | PR 102, step 4 | same | 8/27 -> 10/27; invalid calls 23/83; loops 3; calls 83; median turn 16.8 s | Kept (the refusal names the symbols to declare). `statmech-oscillator-heat-capacity` 0 -> 2 of 3. Five cases still 0/3: ideal gas and fermions (prose question instead of the tool), Ising (the model gave a Boltzmann weight as the bond energy: a verified result about a different system), noncommuting spin and wrong partition function (wrong or no tool). |
| 12 | `false-claim-check-identity-group` | `check_identity` as a keyword group (`02a95b3`) | 16 `false_claim` cases x 3, seed 7 | 33/48 (68.8%) against 31/48 with it as a core tool (34/48 with the stale `false-pythagoras` check corrected); first tool chosen 30/48 (was 35/48); no tool called 9 runs (was 4); calls 55 (was 47); median turn 9.1 s (was 7.9 s) | Moved from core to a keyword group because the offline token budget (5% over the original baseline, already used up to 4.3-4.8% on `main`) has no room for a core schema. The cost is availability: `false-derivative` went 3/3 -> 0/3 because "...is x^x, right?" matched no keyword, so the tool was not loaded and the model answered from memory. |
| 13 | `false-derivative-recheck` | keyword `right?`/`correct?`/`true?` added | `false-derivative` x 3, seed 7 | 0/3 -> 3/3 | Kept. The remaining no-tool runs are the four cases that also fail without this tool (`ode-no-closed-form...`, `integral-theorem-with-a-pole...`, two statmech). Phrasings without any keyword still miss the tool: a known limit of loading by words. |
| 14 | `baseline-2026-11-19-all` | `c7f00a3` | core + reliability, 183 cases x 3, seed 7, `--compute-timeout 10`, routing on | 363/510 (71.2%); repeats 123/122/118 (std 2.16); tool selection 86.1%; valid arguments 88.0%; completion 93.7%; false verified 0; 258 of 798 calls unnecessary; median turn 7.0 s, p95 20.2 s; 4,830 s wall | The reference for this work. Not comparable with row 5 (core only, before PRs 99-102). |
| 15 | `baseline-2026-11-19-heldout` | `c7f00a3` | 6 cases x 3 | 17/18 (94.4%); false verified 0 | Not independent evidence (see row 3); a regression guard. |
| 16 | `baseline-2026-11-19-probe` | `c7f00a3` | frozen probe suite, 38 cases x 3 (prompts digest `bd8f394f3c86`) | 98/114 (86.0%); repeats 32/33/33; false verified 0 | The suite was frozen before any trace of row 14 was opened. Failures are listed in `triage-2026-11-19.md`. |

Rows 1-3 predate the model-identity table (PR 96); rows 5 onward carry it in their reports where noted, and rows 6-8
state the digest `149a0ff8a1ca` unchanged through the run.

## Probe suite (frozen 2026-11-18)

`--suite probe` (`cases_probe.yaml`, 38 cases / 44 turns, not part of `all`): false identities and their true twins,
missing and given assumptions, restrictions across turns, unsupported mathematics stated as a limit and not as an
impossibility, premises that are not proofs, plain-text notation, two-turn explorations, and requests that must be
answered with a question. Alternative valid tools are listed in `also_ok`. Every reference call meets its
`expect_card` against the engine (`--check-cases`).

It was written while the full baseline run was still going, **before any trace or report of that run was opened**, and is
frozen: a test pins the digest of the prompts (`bd8f394f3c86`) and of the whole cases (`6c095266ac4c`). It has the same
author as the fixes that will be measured on it, so it generalises less than the held-out suite (which stays unread and
unused for tuning). A probe that fails is a finding; its prompt is never reworded. If the scoring work later adds an
`accept:` block to a probe case, the cases digest changes and the prompts digest does not; the row that does so says so.

## Planned (not run yet)

- Fresh core baseline x3 on `main` once `check_identity` has merged.
- Control run on the other tool-capable models served by metacon, once, on the cases that still always fail.
- Statistical mechanics, measured as separate steps: deterministic fixes (shape repair, worked examples, chain
  syntax); then the failure-budget change (also reporting model calls and seconds per turn); then the prompt rule
  alone, expected to be reverted.
- Thinking on against off on the cases where the model answers from memory.
