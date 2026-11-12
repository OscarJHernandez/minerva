# Handbook benchmark

`uv run python -m minerva_knowledge benchmark` runs hand-written expression cases through the real
identity finder (retrieval, then the sandboxed engine) and reports whether the right identities are
applicable and the wrong ones are not. Cases live in
`packages/knowledge/src/minerva_knowledge/data/benchmark/identity_cases.yaml` and are written by a
person, independently of any generator. The same cases run in the test suite.

- `expect`: entries that must be APPLICABLE. `forbid`: entries that must not be.
- `pending: true`: the handbook is known to lack the identity (or the matcher to reject the spurious
  one). Pending cases are reported but never fail a run; a pending case that already passes fails the
  test suite until `pending` is removed, so progress is always locked in.

## Results

Numbers are from one machine; latency varies, the counts do not.

| Measure | Baseline | After matcher fix |
| --- | --- | --- |
| Asserted cases passing | 26 / 26 | 32 / 32 |
| Expected identities found (asserted cases) | 23 / 23 | 23 / 23 |
| Pending: expected identities still missing | 13 | 13 |
| Spurious (forbidden) matches shown | 8 | 0 |
| Mean applicable matches per input | 2.33 | 1.42 |
| Latency median / max | 65 ms / 1216 ms | 50 ms / 1199 ms (the max is the first call in a process) |
| Rules in the handbook | 39 (114 entries) | 39 (114 entries) |

The matcher fix (`_binds_fraction_of_multiple` in `packages/engine/src/minerva_engine/ops.py`) stops a
rule written as `n*w` (like `sin(2*a)`) from matching by binding `w` to a fraction of the user's term.
`sin(2x)`, `sin(6x)` and `sin(2x+2y)` still match; `sin(x)`, `sin(3x)`, `sin(x/2)` and
`sin(x+y)` no longer report the double-angle rule. Inputs such as `sin(2x/3)` are conservatively not
offered as double angles.

Baseline observations that motivated the next changes:

- `sin(x)`, `sin(3x)`, `sin(x)sin(y)` and `sin(x/2)` all report the double-angle rule as applicable,
  because the matcher binds `a = x/2` (or `3x/2`) in `sin(2a)`. The rewrite is true but not what a
  reader means by "double angle".
- Triple angle, sum-to-product, `tanh(a+b)`, `(a+b)^3` and similar identities have no entry.
- Four existing rules cannot fire on their own left-hand side by design (see `NOT_SELF_APPLICABLE` in
  `packages/knowledge/tests/test_benchmark.py`).
