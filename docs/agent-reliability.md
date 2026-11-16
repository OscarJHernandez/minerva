# Agent reliability: tool names, notation, evidence, routing and cost

This records one investigation (a chat turn that took 24 s, four model calls, 58 009 input tokens and showed
"formulas with engine evidence 0/12" although the engine had derived the right equations) and what was
changed. Every number below was measured here; where something was not measured, it says so.

## The failure, and what the code showed

The request: the equations of motion of `L = ½ m R² (θ̇² + sin²θ φ̇²) − m g R cos θ`.

| Observation | Cause found in the code | Change |
| --- | --- | --- |
| `mechanics.analyze_lagrangian` was refused | Nothing builds a dotted name. The prompt wrote ``Tools: mechanics. … `analyze_lagrangian` `` and `load_tools` answered `Loaded mechanics: analyze_lagrangian…`; the model copied the pattern. Dispatch is an exact registry lookup, and the refusal listed all 34 tools. | Group names are never written next to a tool name with a period (a test over every group). `toolsets.resolve_tool_name` accepts only closed aliases (`<group>.<tool>` when that group really holds the tool, `functions.<tool>`, case and spacing). Anything else stays unknown and is refused, with the instruction to call a tool by exactly its name. The trace keeps the name the model wrote and how it was read. |
| `theta_dot^2 + sin(theta)^2*phi_dot^2` "has no generalized velocities" | `analyze_lagrangian` always parsed LaTeX. The LaTeX parser reads a run of letters as a product of one-letter symbols: `theta_dot` became `t·h·e·t·a_d·o·t` and `sin(` became `s·i·n`. The engine itself reads text `theta_dot` correctly. | The LaTeX parser now refuses an unbraced multi-letter subscript, a function name followed by `(` without a backslash, and a spelled Greek letter name, each with the corrected form (all operations). `analyze_lagrangian` and `search_invariants` take `format` (`auto`, `latex`, `text`; `auto` picks one of the two strict parsers and never rewrites the input), strip a leading `L =`, and name a declared coordinate that appears nowhere. |
| 0/12 formulas with evidence | The match was a string comparison on lightly normalised LaTeX (`segments._mark`): `\sin\theta` against `\sin{\left(\theta\right)}`, term order and `E` against `h` all missed. The claim checker never saw tool results, and classified every equation with a time derivative as "an equation of motion → unverified". Lone symbols were sent to the checker and counted. | `claim_evidence.py` links a claim to a verified equation by mathematics (below). Lone symbols are not counted; non-relations are not sent to the checker. |
| 58 009 input tokens | About 14 000 estimated tokens per call (system 2 900 + core schemas 5 500 + mechanics rules and schemas 5 400) times four calls; every earlier result of the turn is re-sent in full. | The turn needs one model call when routed (below). The per-call size did not shrink (see "Not done"). |
| 24 s | Engine analysis is 0.5 s warm (parse 5 ms, derive 100, solve 30, check 35, conserved-quantity search 355) and 2.3 s on the first call after start, which `Engine.warm_up` already pays at API startup. Four sequential model calls and one claim check made up the rest. | Fewer model calls; stage timings in the report. The 8.1 s tool time in the recorded trace was **not reproduced** here (0.7 s warm through `execute`); the next debug report will show whether it was the sandbox queue, the host or the model call. |
| Long answers and numerical-integration offers | `run_lagrangian`'s model text ended with an unconditional "next step: integrate_motion" paragraph. | It now says to answer what was asked and that numerical integration only follows a request, with the user's numbers. |

## Evidence: how a model equation is linked to a verified result

Implemented in `packages/engine/src/minerva_engine/claim_evidence.py`, used by `ClaimChecker` before the
relation logic. Three rules, nothing else:

1. **Same left side, equivalent right side** (`\ddot\theta`, `p_\phi`): the right sides agree under
   `verify_equal` (sampled points; a symbolic zero only where the result's own check was symbolic).
2. **Proportional residuals**: `(a − b) = k (c − d)` for a verified `c = d`, with `k` free of the coordinates,
   velocities and accelerations of the system. The zeros of `k` (`m ≠ 0`, `R ≠ 0`) are reported as conditions.
   Covers terms moved across the equals sign and an equation multiplied by parameters.
3. **The model's name for a verified conserved quantity or momentum** (`E = …` against `h = …`). Only for those
   roles: `y = 2` is not supported by a verified `x = 2`.

A match is status `supported` and shows the `from_tool` mark with a note ("same statement as the verified
result r1, in an equivalent form; equivalent when sampled. It holds where …"). It carries the restrictions of the
result. It never raises a claim to "proved": the basis is symbolic only when the tool's check was a symbolic zero
as well. Matrices, vectors, distributions and sums are never linked.

What stays unchecked, deliberately: a wrong sign, an error of 1e-6, a different conserved quantity, a plain
renaming, and a true form multiplied by a factor that depends on the coordinates (`sin θ · θ̈ = …`). A wrong
equation is **not** shown as contradicted (a false statement about a system Minerva was not asked about cannot be
told from a different system), it is simply not backed. All of these are tests
(`packages/engine/tests/test_claim_evidence.py`, `services/api/tests/test_claim_evidence_agent.py`).

Coverage is narrower than "every correct statement": any form that is neither a restatement of the same left
side nor proportional by a constant factor is not linked, and the evaluation reports exactly that.

## Routing

`agent/routing.py`: a message with one formula that reads as a Lagrangian with unambiguous velocities
(`\dot{q}` or `q_dot`) **and** the words "equations of motion" or "Euler–Lagrange" around it runs
`analyze_lagrangian` before the first model call. Either signal alone never routes; indexed coordinates
(`\dot{q}_1`), accelerations, several formulas and requests that mention the Hamiltonian are left to the model.
The call is executed like any other (validation, sandbox, verification); if it does not come back verified the
route is dropped and the model plans the turn. The model still writes every word of the answer. Routed steps
cost no model call, are marked in the trace, timeline and log (`agent.routed`), and
`MINERVA_AGENT_ROUTING=false` turns the rule off. It is one rule on purpose; the list in `routing.RULES` is where
the next would go.

## Measurements (offline suite, same trajectories before and after)

`scripts/eval_offline.py` runs the real loop, tools and claim check with a replayed model
(`docs/benchmarks/orchestration-offline.md`; baseline: the same trajectories on the code before this work,
`orchestration-offline-baseline.json`). Tokens are estimated with the pessimistic `budget.estimate_tokens`; the
recorded failure replays to 57 239 against the 58 009 the host reported.

| Spherical pendulum | Model calls | Tool calls (ok) | Est. input tokens | Equations backed |
| --- | --- | --- | --- | --- |
| Recorded failure, before | 4 | 3 (1) | 57 239 | 0 of 4 |
| Same model behaviour, after | 4 | 3 (2) | 59 866 | 4 of 4 |
| Model calls the tool itself with text syntax (no routing), after | 2 | 1 (1) | 28 990 | 4 of 4 |
| Routed, after | 1 | 1 (1) | 14 923 | 4 of 4 |

So the supplied example needs one successful tool call and one model call, with ≈74 % fewer estimated input
tokens than the recorded run (the test requires at least 50 %). That is a change of *behaviour* (routing removes
three of the four calls), not a smaller request: **the per-call prompt is about 4 % larger now** (longer group
headings, a `format` field, the restriction line). The suite flags any trajectory whose tokens grow by more than
5 % as a regression.

Across all 16 trajectories (algebra, calculus, Lagrangians, conserved quantities, matrices, ODE, vector
calculus, notation variants, malformed and unreadable calls, an unknown tool): 8 of 16 ended correct before,
16 of 16 after; claims without backing 30 → 6, none of the 6 a wrong statement presented as checked.

Latency, measured on this machine (one run, noisy, shared CPU): tool call 0.7 s warm; claim check of four
equations about 1.0 s (it was 0.7 s before, but linked nothing). Timings are reported in the suite and never
gated.

## What the offline suite cannot show

How a real small model behaves. It replays given model actions; whether a model *chooses* `analyze_lagrangian`,
avoids dotted names now that the prompt no longer shows them, or recovers from a refusal is only measurable with
`scripts/eval_models.py` against a host. **That live comparison has not been run for this change**; run
`make eval-reliability` style suites (`--suite reliability`, `--repeat 3`) before and after on the target model
and compare with `--compare`.

## Not done, and why

* Per-call prompt size was not reduced. The largest schemas (`find_conserved_quantities`, `perturbation_solve`,
  `integrate_motion`, `creative_telescoping`, `integrate`: 2.4–3.5 k characters each) carry safety wording that
  must not be trimmed without a model to test the effect on; the debug report now shows the size by source
  (system, tools, history, this turn) so that work can be measured.
* The 8.1 s tool time was not reproduced; see above.
* Distinct verification marks (proved / numerically checked / under conditions) were not added: the existing six
  marks and the new evidence note already say which, and the split touches the UI, scoring and the debug report
  for a presentation change. The `basis` is available in the claim debug record.
* Only Lagrangian requests are routed.
* `format` exists for `analyze_lagrangian` and `search_invariants`. Other Lagrangian tools
  (`expand_motion`, `perturbation_solve`, `integrate_motion`, `find_conserved_quantities`, `time_derivative`) take
  LaTeX only, and now refuse text syntax with the fix instead of misreading it; the engine does not support text
  for them.

## Argument audit

Every main expression field of every tool must state its syntax (a test). Syntax by tool family:

| Tools | Expression syntax | `format` argument |
| --- | --- | --- |
| `simplify`, `differentiate`, `integrate`, `series`, `substitute`, `rewrite_part`, `find_identities`, `isolate_variable`, `solve_polynomial`, `check_units`, `sums`, `expectation`, `distribution`, `creative_telescoping`, `approximation`, `variation` | LaTeX | none (text is refused with the fix) |
| `analyze_lagrangian`, `search_invariants` | LaTeX or text (`theta_dot`) | `auto` / `latex` / `text` |
| `expand_motion`, `perturbation_solve`, `integrate_motion`, `find_conserved_quantities`, `time_derivative` | LaTeX | none |
| `matrix_calculus` | LaTeX; text for the gradient actions | per action |
| `vector_calculus`, `electromagnetism` | LaTeX (explicit components: text) | none |
| `ode_*` | the user's notation, `format` `latex` or `text` | `format` |
| `tensor_canonicalize` | text (`S(a,b)*A(-a,-b)`) | none |
