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
| 17 | `exp1-spelling-calculus` | `fix/formula-spelling` | the 33 `calculus` cases x 3, seed 7, compared with row 14 | 65/99 -> 68/99 (repeats 23/21/21 -> 22/24/22); targeted case `integrate-definite` 0/3 -> 3/3; other changes are flaky cases (`delta-integral-uses-the-tool` 2->1, `differentiate-mixed` 3->2, `differentiate-first` 1->2, `vector-curl-curl-rewrite` 2->3, `homepage-stokes-curved-surface` 0->1, `vector-premise-is-conditional` 3->2); tool calls 176 -> 192; false verified 0 | Kept on the targeted case, not on the total: +3 is inside the noise (about 4). Hypothesis (a correct restated integral is unchecked only for its spelling) confirmed by the one case it predicted; the normalisation only equates spellings and its wrong twins stay different (`test_formula_spelling.py`). |

| 18 | `exp2-matrix-size` | `fix/matrix-redundant-size` (not merged) | `tensor_matrix` + `physics` cases x 3, seed 7, compared with row 14 | 95/144 -> 95/144 (repeats 32/34/29 -> 31/32/32); targeted failure (entries plus a redundant size) not visibly reduced; `matrix-merge-needs-commutation` 3 -> 0, `matrix-evolution-asks-for-the-span` 0 -> 2, other moves are flaky cases; false verified 0 | **Reverted** (branch kept, not merged): no change in the total and the one large drop is unexplained, so it is not shown to help. The 23 refused calls it targeted were a small share of failures; the cause there is the model's choice of tool, not the size field. |
| 19 | `exp3-explicit-operation` | `feat/route-explicit-calculus` | `calculus`, `symbolic`, `unsupported`, `ambiguous`, `false_claim` cases x 3, seed 7, compared with row 14 | 185/270 -> 192/270 (repeats 64/60/61 -> 65/66/61); targeted cases `differentiate-first` 1 -> 3, `unsupported-derivative-abs` 0 -> 3, `unsupported-series-pole` 0 -> 3; regressions are flaky cases (`distributional-wrong-normalisation-is-rejected` 3 -> 2, `statmech-divergent-series-is-refused` 2 -> 1, `isolate-unsupported-is-not-answered` 1 -> 0); tool calls 354 -> 377, no-tool runs 38 -> 25; false verified 0 | Kept: +7 is at the edge of the noise (about 4-5), but three cases the rule was written for moved 0-1 -> 3 and no-tool runs fell by a third. The rule declines on `=`, two formulas, extra words, `@` references and any matrix, vector or tensor conversation (`test_routing_explicit_operation.py`). |

| 20 | `think-on-nocall` | thinking requested, nothing else | the 11 cases with no tool called in row 14 x 3, seed 7, model `minerva` (gemma4:26b build, digest `149a0ff8a1ca`), host default reply limit 1200 | thinking off 1/33 -> on 9/33; no-tool runs 29 -> 19; median turn 4.1 s -> 13.7 s; output tokens x5.6; 7 runs cut off at 1200 tokens; false verified 0 | Not adopted. A real effect on these cases (`handbook-constant`, `unsupported-series-pole` 0 -> 3, `ode-no-closed-form...` 0 -> 2), but three times the latency and a reply budget the host default does not give. Measured fairly in the model comparison (docs/reliability/model-comparison.md). |

| 21 | `model-comparison-stage1` | 7 local models, thinking off and on (`gemma4:26b` deployed build served as `minerva`, `gemma4:26b`, `qwen3.5:35b`, `qwen3.6:27b`, `qwen3.8:27b`, `glm-4.7-flash`, `granite4.2:30b`, `mistral-small3.2:24b`) | probe suite (38 cases) x 3, seed 7, reply limit 4096 | 15 configurations; best `qwen3.5:35b` thinking on 106/114 (93.0%), deployed build 101/114 (88.6%), granite on 90/114; thinking helps the qwens (+7 to +9), is neutral for gemma and glm, hurts granite (-8); `mistral-small3.2:24b` 69/114 | A screen, not a ranking (see row 22). `magistral:24b` stopped at 32 of 114 runs and `ministral-3:14b` never run, by the maintainer's decision. Digest, thinking state and reply budget on every row: `docs/benchmarks/model-comparison-2026-11-20/`. |
| 22 | `model-comparison-stage2` | `qwen3.5:35b` thinking on, the deployed build thinking off (reference), `glm-4.7-flash` thinking off | full suite (510 runs) and held-out (18) x 3, seed 7, reply limit 4096, code `a295cb82f7` | 389/510 (76.3%), 372/510 (72.9%), 316/510 (62.0%); `qwen3.5:35b` ahead of the reference beyond the noise (5.7 against a limit of 4.3 passes per repeat) with tool selection 95.3% against 87.3% but valid arguments 83.5% against 88.0%; `glm-4.7-flash` loops (68 runs) and fell from 91.2% on the probe suite; false verified 2 for `qwen3.5:35b` (the A-009 defect, see row 23) | The screen over-predicted for `glm-4.7-flash`. `qwen3.5:35b` is the most accurate as measured and not eligible under the rule until its full suite is re-run on the fixed code; the one case re-run (5 repeats, `main` at `ab50f75`): 5 passed, false verified 0. Full write-up and recommendation: `docs/reliability/model-comparison.md`. |
| 23 | `defects-found-by-the-comparison` | n/a (fixes, not experiments) | A-008 (harness did not give the claim checker the declared matrices in single-turn cases), A-009 (`check_identity`, `rewrite_part`, `substitute`, `sums` and the Lagrangian tools read a declared matrix as commuting numbers), A-010 (a redefinition left stale definitions; a division by zero counted as disagreement), V-017 (a clock-cut probe search passed without saying so) | PRs 117, 121, 123, 124, 125, each with tests that fail on the old code; `mistral-small3.2:24b` re-run on the fixed harness: false verified 0 | None of the three false-verified counts was a model failing; two were our defects. The V-017 verdict is unchanged and the question of whether a clock-cut pass should be `not_verified` is open. |

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
