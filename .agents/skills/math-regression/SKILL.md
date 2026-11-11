---
name: math-regression
description: Add or change a Minerva mathematical operation, parser rule, tensor feature or handbook (knowledge-base) entry and prove it correct with regression tests. Use whenever work touches packages/engine, packages/knowledge or the compute endpoints, or when adding identities, integrals, series, physics equations or constants.
---

# math-regression

Minerva's rule: **correctness > simplicity > maintainability > features.** A result is shown only if it
was verified; otherwise the status says why. Follow this checklist and stop to ask if any step is unclear.

## 1. Decide which kind of change it is

| Change | Where | Verification level to declare |
| --- | --- | --- |
| New identity the engine may apply | `packages/knowledge/src/minerva_knowledge/data/entries/*.yaml`, `rule:` | `symbolic` only if `python -m minerva_knowledge verify` reports a symbolic zero; otherwise `numeric` |
| Derivative / antiderivative | same, `check: {kind: derivative\|antiderivative}` | `symbolic` (differentiation) |
| Definite integral / series | same, `check: {kind: integral\|series}` with `params` ranges | `numeric` |
| Tensor symmetry | same, `check: {kind: tensor}` | `structural` (only monoterm symmetries) |
| Physics equation | same, `check: {kind: equation, units}` | `dimensional` (units only: say so in notes) |
| Literature fact the engine cannot check | same, `verification: stated` + `statement_latex` | `stated` (never applied) |
| New engine operation | `packages/engine/src/minerva_engine/ops.py` + `service.py` + API route | explicit statuses, independent verification |

Never claim a stronger level than the checker establishes. Never copy text or tables from books: write the
formula yourself and cite by locator (`DLMF §4.21`).

## 2. Write the entry or operation

- State **validity conditions** as `conditions` (`positive`, `real`, `nonnegative`, `integer`, ...). A rule
  like `log(a*b) = log(a) + log(b)` must carry `a`, `b` positive: SymPy symbols are complex by default.
- Pattern variables are declared in `variables`; every symbol in `lhs`/`rhs` must be declared.
- Text syntax reserves `E`, `I`, `pi`, `oo`: do not use them as variable names (use `En`, `Ef`, ...).
- SymPy evaluates some things while parsing (`sqrt(x**2)` with `x` real becomes `Abs(x)`, `exp(x)*exp(y)`
  merges). A rule whose `lhs` auto-evaluates will never match; pick a different formulation.

## 3. Prove it

```bash
uv run python -m minerva_knowledge verify          # every entry at its declared level
uv run pytest packages/knowledge packages/engine   # schema, retrieval, find-identities, parsers, sandbox
```

Add tests next to the change:

- **Positive:** the expected result with the exact expected string/LaTeX.
- **Negative:** things that must **not** simplify or apply (e.g. `\sqrt{x^2}` is not `x` without assumptions;
  `\log(xy)` is not split without positivity).
- **Assumption ladder:** the result changes with `real` / `positive` / `nonnegative`.
- **Honesty:** an input the backend cannot handle returns `unchanged` / `not_verified` / `unsupported` /
  `parse_error` / `resource_limit`, never a plausible guess.
- **Safety:** hostile inputs (huge exponents, deep nesting, `factorial(10**6)`, attribute access) are
  rejected statically or stopped by the sandbox without crashing the server.

## 4. Review checklist

- [ ] Verified result only: no code path returns an unverified candidate.
- [ ] The match filter still rejects matches that invent roots, reciprocals or `I` (`_invents_structure`).
- [ ] No `eval` / `exec` / `sympify` / `parse_expr` on user input.
- [ ] New operations run through `Engine` (sandbox) and are registered in `ops.OPERATIONS`.
- [ ] Tests assert the explicit status for failure cases.
- [ ] Docs updated if behaviour or conventions changed (`AGENTS.md`, `docs/architecture.md`).
