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

| Measure | Baseline (before any change) |
| --- | --- |
| Asserted cases passing | 26 / 26 |
| Expected identities found (asserted cases) | 23 / 23 |
| Pending: expected identities still missing | 13 |
| Pending: spurious (forbidden) matches shown | 8 |
| Mean applicable matches per input | 2.33 |
| Latency median / max | 65 ms / 1216 ms (the max is the first call in a process) |
| Rules in the handbook | 39 (114 entries) |

Baseline observations that motivated the next changes:

- `sin(x)`, `sin(3x)`, `sin(x)sin(y)` and `sin(x/2)` all report the double-angle rule as applicable,
  because the matcher binds `a = x/2` (or `3x/2`) in `sin(2a)`. The rewrite is true but not what a
  reader means by "double angle".
- Triple angle, sum-to-product, `tanh(a+b)`, `(a+b)^3` and similar identities have no entry.
- Four existing rules cannot fire on their own left-hand side by design (see `NOT_SELF_APPLICABLE` in
  `packages/knowledge/tests/test_benchmark.py`).
