# Result presentation and reuse audit

Reviewed against `165fcdda4044a31b72883f168b53505887c4a38a`. This is a focused review of the
path from a mechanics tool result to chat display and reuse, not a complete correctness audit.

## Fixed in this change

Tracked in [#147](https://github.com/OscarJHernandez/minerva/issues/147) (visible qualifications) and
[#148](https://github.com/OscarJHernandez/minerva/issues/148) (readable derivative cards). Both are
implemented in one pull request; whether a given deployment has it depends on the deployed build.

- **The useful result was missing from the card.** `DifferentiateResult` on the engine includes
  `on_motion_latex`, `zero_on_motion`, `symbolic_zero_on_motion`, and `definitions_used`; the browser
  type and derivative panel omitted all four. The card now leads with the checked zero, explicitly
  scoped to the supplied equations of motion, and exposes the definitions separately.
- **Two different checks were presented as one.** A total derivative can pass its independent numerical
  derivative check and separately be a symbolic zero on the equations of motion. The old panel's
  blanket “no independent symbolic test” text obscured the second fact. The new view names each check.
  A nonzero substituted expression remains explicitly described as not separately sampled.
- **A substitution was labelled as another derivative step.** The engine appends the on-motion
  evaluation to `steps`; taking `steps.slice(0, -1)` then showed the raw result as an intermediate
  derivative. Total-derivative cards now distinguish the unrestricted formula and on-motion evaluation.
- **Oversized centered math could hide its beginning.** The KaTeX display scroller now contains a
  content-width inner formula, so the leading term and the end are both reachable. Long derivatives
  are collapsed at normal font size, with the original LaTeX retained for display and copying.
- **Inherited qualifications depended on model prose.** `loop.py` attaches
  `inherited_restrictions_latex` and `unverified_inputs` to card data, but the frontend had no readers
  for them. `CardView` now renders these qualifications for every card type (`provenance-notes.tsx`), in three
  kinds: inherited conditions, an equation the model wrote and Minerva never verified (checking the
  computation does not establish it), and a list component whose original tool verification and
  restrictions were not carried forward (it does not say the tool never verified it). The split reads
  one suffix, pinned by `packages/ui/fixtures/input-provenance.json` and a test against `refs.py`. This does
  not change the engine verdict or promote an unverified input.
- **A summary can change what a reference appears to name.** A card number and generic copy buttons
  refer to `result_latex`, not `on_motion_latex`. They now stay beside the unrestricted derivative in
  its disclosure; they are not moved onto the zero summary or silently made to copy zero.

## Follow-up priorities

| Priority | Finding and code path | Consequence / next step |
| --- | --- | --- |
| High | Formula provenance and vector inference, tracked in [#146](https://github.com/OscarJHernandez/minerva/issues/146): `agent/segments.py`, `declarations.py`, `loop.check_model_claims` | Harmless Greek trig formatting loses provenance, and a sentence mentioning a vector can contaminate scalar claims. Fix conservative normalization and declaration scope together, with wrong-formula and real-vector negative cases. |
| High | `agent/equations.py` skips formulas longer than `MAX_EQUATION_CHARS` (500); `agent/refs.py` excludes `time_derivative` from `REFERENCEABLE_TOOLS` | Tracked in [#149](https://github.com/OscarJHernandez/minerva/issues/149). The longest total derivatives have neither an equation handle nor a result handle, precisely when copying is least reliable. Introduce bounded references to stored expressions without weakening parser/size limits or confusing a general derivative with its scoped zero. |
| High | Listed equation reuse deliberately carries `UNLINKED_INPUT` in `agent/refs.py` | The original row's restrictions and evidence are not automatically linked. The new warning makes this visible; typed per-row provenance remains needed. Coordinate with [#133](https://github.com/OscarJHernandez/minerva/issues/133), rather than merely adding more equation numbers. |
| Medium | `time_derivative.py` only proposes `simplify(raw)` below `TIDY_MAX_OPS = 120` | An algebraic relation can expand into a huge derivative even when a compact result exists. Profile a bounded compact-candidate path, recheck it against the original expression, and retain the verified raw candidate if compaction fails. Do not simplify in the browser or bypass numerical verification. |
| Medium | `agent/evidence.py` produces one evidence record per tool, while discovery and on-motion evaluation have multiple conclusions | A numeric top-level record does not capture every symbolic sub-result. Store scoped evidence per conclusion; never relabel a whole card “proved” because one sub-result has a symbolic zero. |
| Medium | `scripts/replay_trace.py:replay_turn` restores earlier turns as plain text | Tracked in [#150](https://github.com/OscarJHernandez/minerva/issues/150). A replay of a follow-up using stored constant definitions or prior evidence is not equivalent to a real chat continuation. Rebuild the same stored-card context as the chat route and test a two-turn case. |
| Medium | `engine` result schemas and browser interfaces are maintained separately | Fields can disappear from presentation without a TypeScript error, as happened here. `services/api/tests/test_web_engine_contract.py` now pins the derivative result (every engine field must be declared in the browser type); other result shapes still need the same, with explicit semantics for absent/null/false values. |

## Validation boundaries

The UI renders only fields already returned by the engine. It does not create a shorter formula, decide
functional independence, or prove a new identity. Browser checks use the affected run's long derivative
in a temporary local fixture, with an isolated mocked account response; they are not a live model or
API test. The raw report and temporary fixture are not repository artifacts.

The broader audit findings above are code-path observations. The first-turn replay against the affected
build completed with `whole-turn replay: same`, including the recorded formula marks, using the offline
`ReplayLLM` and Minerva's own sandbox. The outer OS sandbox initially prevented its child process from
running; the successful replay ran after that environment restriction was resolved. This establishes
reproduction of the first turn, not the interrupted attempt or the follow-up context restoration.

Validation is described in the pull request. Browser checks use a temporary local fixture with an
isolated mocked account response, not a live model or API. No backend behaviour changed.
