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

One more outcome is neither: the model re-types a verified constant under the tool's own name with another
expression (`C_4 = … r^2 \dot r\,\ddot\theta …` for `\dot\theta`, seen in issue #132). If the left side is
exactly the name of a verified conserved quantity or momentum and the right side provably differs (a gap
confirmed at 60 and 120 digits), the claim stays `unchecked` and its note names the verified equation
(`evidence_rule: "mismatch"`). Still not "contradicted": the mark says the equation was not backed, and why it
should be compared. A different name on the left, or evidence from a result that set no role, never mismatches.

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
`MINERVA_AGENT_ROUTING=false` turns every rule off. `routing.RULES` lists them: after the Lagrangian rule came
`followup_reuse`, `ode_analyze`, `matrix_simplify` and (2026-11-19) `explicit_operation`, each added because a live trace showed the model not making
the call (`docs/reliability/triage-2026-11-18.md`). A routed call is resolved (`@rN`), repaired, cleaned and run in the
per-call tool context exactly like a model call, and a rule names the statuses it keeps as the answer: a non-commuting
matrix product that comes back `conditions_not_satisfied` is the answer, an ODE the analyser refuses is not.
The rules never fill in what the user did not write (variable, bounds, sizes, commutation, invertibility,
assumptions) and decline when a message is ambiguous (two results in the last turn, two variables named, bounds
in words, a method named for an ODE).

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
* Routed: Lagrangian equations of motion, `differentiate`/`integrate` of "that result" (`followup_reuse`), `ode_analyze` of a bare equation and `matrix_simplify` over matrices the user declared n x n. Not routed, deliberately: simplify, differentiate and integrate of a formula in the message (the 2026-11-18 baseline shows the model choosing them correctly), vector calculus and distributions (a coordinate system, declared fields and the delta notation are the user's to state). See `docs/reliability/triage-2026-11-18.md`.
* `format` exists for `analyze_lagrangian`, `search_invariants` and `find_conserved_quantities` (visible; the last one
  since issue 131: the engine reads the Hamiltonian in text too) and, hidden, for the formula tools listed
  in the audit below. Other Lagrangian tools (`expand_motion`, `perturbation_solve`, `integrate_motion`) take LaTeX only: their engine entry points read LaTeX, so a text Lagrangian is refused
  with the fix there (see "Second pass").

## Argument audit

Every main expression field of every tool must state its syntax (a test). Syntax by tool family:

| Tools | Expression syntax | `format` argument |
| --- | --- | --- |
| `simplify`, `differentiate`, `integrate`, `series`, `time_derivative`, `find_identities` | LaTeX; a text form (`sin(x)`, `x*y`) that LaTeX refuses is re-read as text by the loop | hidden `format` (`latex` default, `text`), switched by the recovery policy |
| `substitute`, `rewrite_part`, `isolate_variable`, `solve_polynomial`, `check_units`, `sums`, `expectation`, `distribution`, `creative_telescoping`, `approximation`, `variation` | LaTeX | none (text is refused with the fix) |
| `analyze_lagrangian`, `search_invariants`, `find_conserved_quantities` | LaTeX or text (`theta_dot`, `Xdot`) | `auto` / `latex` / `text` |
| `expand_motion`, `perturbation_solve`, `integrate_motion` | LaTeX (spelled velocities of a declared coordinate are refused with the fix in LaTeX) | none |
| `matrix_calculus` | LaTeX; text for the gradient actions | per action |
| `vector_calculus`, `electromagnetism` | LaTeX (explicit components: text) | none |
| `ode_*` | the user's notation, `format` `latex` or `text` | `format` |
| `tensor_canonicalize` | text (`S(a,b)*A(-a,-b)`) | none |

## Second pass: the input layer, typed diagnostics and recovery

The first pass made the tool names and the notation of one tool dependable. The recorded turn that prompted this one
shows what remained: a model picked `analyze_lagrangian` correctly and wrote the velocities of a block on a wedge as
`Xdot`, `Ydot`, `sdot`; the engine did not know them (`unsupported`, later `parse_error`); the retry arrived with
markdown-escaped products (`\*`); the third call repeated the first; 4 model calls, 3 tool calls, 18.6 s, no formula with
engine evidence. The mathematics was well within the engine's reach (the accelerations are `X'' = 0`, `Y'' = -g`,
`s'' = 0`, derived by hand in the tests); only the spelling was not.

### Root causes, from the code

| Cause | Where |
| --- | --- |
| Derivative spelling was decided in five unrelated places, each with its own closed list (`f"{q}_dot"`, the ODE reader's regexes, the variation jets, the router's regexes, the format marker) | `lagrangian.derive_expr`, `time_derivative`, `ode/reading.py`, `variational/jets.py`, `agent/routing.py`, `tools._TEXT_MARKERS` |
| A symbol that was not a coordinate or a velocity silently became a parameter, so `Xdot` surfaced later as "no velocity" | `lagrangian.derive_expr` |
| Transport damage reached the strict parsers: one `\*` made the whole text LaTeX | `resolve_syntax`, the parsers |
| Failures were prose: the loop could not tell "re-spell this" from "this is unsupported" | `ParseError`, `ops._failure`, the report models |
| A call that differed from an earlier one only by `\*`, a `$` wrapper, `L =` or a default was a new call | `loop._call_key` (raw JSON) |
| No budget per tool, no notion of progress between failures | `MAX_UNPRODUCTIVE`, `MAX_REPEATS` only |
| Three hand-kept lists of which arguments are formulas, none complete | `refs`, `tools`, `loop` |
| The model was never told the numbers of the formulas it had been given, so it copied them | `routers_chat` |

### What was built

Everything is a closed list with a test and an inverse; nothing rewrites a meaning, and nothing touches verification.

1. **One derivative reader** (`minerva_engine/notation/derivatives.py`). Spellings of the derivative of a *declared*
   coordinate are generated from parts (`Xdot`, `X_DOT`, `dX_dt`, `dot_X`, `X_prime`, `X dot`...), the keyword is
   case-insensitive and the stem exact. A name with two readings, two spellings of one derivative in one expression,
   a name that is itself declared, a name with declared assumptions or a velocity of an undeclared coordinate is
   refused with a typed code and the candidates. In LaTeX a letter run such as `Xdot` is a product, so it is refused
   with the fix (`\dot{X}`), never reinterpreted. Used by `analyze_lagrangian`, `time_derivative`, and through them
   every tool that derives equations of motion; routing finds coordinates with the same generator.
2. **Transport cleaning** (`notation/transport.py`, `agent/mathinput.py`): markdown escapes, `$` and code-fence
   wrappers, doubled backslashes, Unicode look-alikes, HTML entities, sentence punctuation, a leading `L =`/`H =`,
   Greek and dotted letters in the syntax of the tool. Runs on every formula argument after `@rN` references are filled
   in and before the call is keyed. In a field read as text syntax only, a braced exponent is the parenthesised one
   (`x^{-1}` -> `x^(-1)`, `(a+b)^{n+1}`; `transport.read_braced_exponents`: directly after `^`, balanced braces, nested
   groups handled; a group with a top-level comma, an empty or an unbalanced one stays refused, and no set literal is
   ever accepted; LaTeX is never touched). The same function runs in the text parser itself, because geometry
   metric entries and the workbench reach it without this layer (issue 131: `(1-2*M/r)^{-1}` was `unsupported syntax: Set`).
   It is not in `clean_transport` (that one is format-independent and `^{12}` is the syntax in LaTeX), so its inverse is
   the corruption operator `syntax:braced_exponent`, not a `DAMAGE` row.
3. **Shape repair** (`agent/arguments.py`): a list sent as a string, one equation where a list is wanted, JSON sent as
   text, a formula under `expression`, a number where text is wanted.
4. **Typed diagnostics** (`notation/models.py`): `InputDiagnostic` (closed `code`, `field`, `symbols`, `suggestions`,
   `recoverable`) on every report that reads a formula; the model sees one JSON line, the log only the code
   (`reason_code`). `InputReading` says what was changed on an accepted input; the card shows "Read as:" and the model
   an `input:` line.
5. **Recovery policy** (`agent/recovery.py`): `FailureClass`; one re-run with the other strict parser when a parse error
   says the text is of the other syntax (`policy: format_switch`); a canonical call key (schema defaults filled,
   numbers and text unified, `format: auto` resolved) so an equivalent call that ran or failed is not run again
   (`refused_equivalent`); a budget of consecutive failures per tool (`budget_exhausted`); a failure with a diagnostic
   code not seen before for the tool is progress. Unsupported mathematics is an answer and is never retried with
   equivalent input. A `resource_limit` result gets a fixed hint (`tools.RESOURCE_LIMIT_HINT`; for `expand_motion`,
   `order: 1` is the smaller call when the user asked for small oscillations) and the turn ends with a summary
   even at the step limit (issue 137).
6. **Handles**: the model is told the handles (`@e4`) of the formulas the user displayed, so it passes the handle and
   Minerva substitutes the user's exact text. The user's own formula is never marked verified by being used.

### Measurements

All numbers were measured here, offline (a replayed model, the real loop, tools and engine). `docs/benchmarks/`:
`orchestration-offline-main-421998e.json` is the stage-0 report of the unchanged code, `orchestration-offline.json`
the current one, `corruption.json` and `corruption-without-cleaning.json` the corruption test.

| The recorded wedge turn | Model calls | Tool calls (ok) | Est. input tokens | Equations backed |
| --- | --- | --- | --- | --- |
| Recorded script, unchanged code | 4 | 3 (0) | 57 164 | 0 of 4 |
| Same script, now (the first call succeeds; the next two are refused as equivalent) | 4 | 3 (1) | 49 393 | 4 of 4 |
| The model writes `Xdot` and gets the analysis at once | 2 | 1 (1) | 29 170 | 4 of 4 |
| `format` stated wrongly (`latex`), re-read as text by the loop | 2 | 1 (1) | 29 198 | 4 of 4 |
| Coordinates as one string, formula under `expression`, `\*` in the text | 2 | 1 (1) | 29 240 | 4 of 4 |
| The Lagrangian typed with `Xdot` is routed | 1 | 1 (1) | 15 052 | 4 of 4 |

The suite now has 25 trajectories (17 before); all end mathematically correct (16 of 17 of the earlier ones did before
this pass; the wedge was added as a failing one first). Claims without backing across the suite: 10 → 6, none a wrong
statement presented as checked. Tokens per trajectory moved by at most +1.3 % (the diagnostics add lines only when
something was adjusted); no trajectory is flagged by `--compare`.

**Corruption test** (`make eval-corruptions`): every known model mistake applied to the 133 reference calls of the
benchmark (33 tools), through the model's own pipeline. 737 corrupted calls: **730 recovered without a model call
(99.1 %)**, 7 refused with a typed reason, **0 different verified results, 0 untyped failures**. Without the cleaning
layer, on a sample of 222 calls that all go through the engine: 26 recovered (11.7 %), 65 refused with a typed reason,
131 untyped failures (the velocity spellings are still read there: that reader is in the engine, not in the cleaning). The
operators are the inverses of the cleaning rules, so this rate says the rules undo what they were written to undo; it is
not an estimate of how often real models make these mistakes (that is what `scripts/diagnostic_codes.py` on a real log
tells), and it is not a measure of the model.

### What this cannot show, and what is not done

* **No live model was run.** Whether a small model passes `Xdot`, follows a `suggestions` list, or uses a handle is
  measurable only with `scripts/eval_models.py` against a host; the targets of the brief (valid arguments, recovery
  from known notation errors, repeated identical failed calls, incorrect verification claims) are measured here
  offline only where a replayed model can show them: repeated identical failed calls are 0 by construction
  (the call key), incorrect verification claims did not change (no check was touched).
* **Function notation is not read**: `x(t)`, `diff(x(t), t)`, `\frac{d}{dt}x(t)` and differentiating with respect to
  `x(t)` or `\dot{x}` are not implemented. In text syntax `x(t)` is still an error and in LaTeX it is still the
  product `x t` unless the tool declares functions.
* **The ODE and variational readers keep their own derivative spellings**; `ydot` is read only where the Lagrangian
  reader is used.
* `expand_motion`, `perturbation_solve`, `integrate_motion` read LaTeX only (a spelled velocity in LaTeX is refused
  with the fix). `find_conserved_quantities` takes `format` since issue 131 and refuses more than three coordinates with
  the typed code `too_many_coordinates` (the limit, what a user may do by hand, and that reduction and a larger search
  are not available yet); the coordinate cap of its schema is 8 so that this reason, not a generic length error, is shown.
* A coordinate named `t` clashes with the default time symbol `t`: `analyze_lagrangian` then answers `not_verified`
  (the Euler-Lagrange check cannot separate the two). The `time` argument says so and the reliability case
  `schwarzschild-chat-conserved-quantities-are-scoped` uses `time: s`. A typed refusal for the clash is not built.
* Inline formulas have no handle (only displayed ones); distinct verification marks and the per-call prompt size are
  still open (see "Not done" above).
* A coordinate named like a derivative spelling (`xdot` with `x`) is kept as declared, but the LaTeX printer shows a
  symbol named `xdot` as `\dot{x}`: name coordinates so that this cannot happen.

## Equation labels inside a formula

The server numbers equations; the model is told not to, but it copies the `$$...$$ (3)` pattern from the replayed
history and sometimes writes the number inside the formula (`$$E = mc^2 (1)$$`). The key of the formula then contained
the label, so the same formula restated without it was a new equation (issue 134: one cycloid under (2), (3) and (7)).
`strip_label` reads one label off the very end of a displayed formula, by who wrote it:

| Form | model reply | user message | tool output |
| --- | --- | --- | --- |
| `\tag{3}`, `\tag*{3}` | removed | removed from the key | kept |
| `\quad`, `\qquad`, `\hfill`, `\hspace{..}` then `(3)` | removed | removed from the key | kept |
| `\text{(3)}`, `\mathrm{(3)}` | removed | removed from the key | kept |
| whitespace then `(3)` | removed unless it can be a factor | kept | kept |

A bare ` (3)` is kept after a command name (`\sin (2)`, `\Gamma (3)`), an operator, relation, comma or opening bracket,
a letter the same formula applies as a function elsewhere, and when removing it would unbalance brackets. `f(1)`,
`(1+x)`, `2(3)`, limits and matrix entries are never touched. A formula already numbered as written is found first and
never stripped. Accepted risk: a formula new to the conversation such as `y = g (1)` loses its `(1)`; every removal is
recorded. Not handled: labels inside `align`, `aligned`, `gather`, `eqnarray`; a tool result that ends in `(N)` quoted
before its card.

## The three-dimensional delta and the model

The model reaches `distributional` only through `vector_calculus` (a keyword group and one rule block); the coefficient,
the sign, the source and the verdict on a user's statement come from the engine, which has no argument for any of them.
`known_latex` carries the engine's printed equation, and `segments.normalize` treats `\mathbf r`/`\mathbf{r}` and
`\delta^3(...)`/`\delta^{(3)}(...)` as one spelling so a correctly written formula is not shown as unchecked for its
spelling; a `\delta^{(3)}(...)` is never sampled by the claim checker. Vector and distribution formulas are still never
linked by mathematics (an equation about a delta is marked verified only by an exact match with what the tool printed).
The offline trajectory `vector-distributional-divergence` and the eight `distributional-*` cases cover this; the first
real-model measurement is recorded in the homepage evaluation.
