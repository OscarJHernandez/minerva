# Handbook benchmark

`uv run python -m minerva_knowledge benchmark` runs hand-written expression cases through the real
identity finder (retrieval, then the sandboxed engine) and reports whether the right identities are
applicable and the wrong ones are not. Cases live in
`packages/knowledge/src/minerva_knowledge/data/benchmark/identity_cases.yaml`, separate from the
generator that adds handbook entries. The same cases run in the test suite.

- `expect`: entries that must be APPLICABLE. `forbid`: entries that must not be.
- `pending: true`: a known gap (a missing identity, or a spurious match not yet rejected). Pending cases
  are reported but never fail a run. A pending case that already passes fails the test suite until
  `pending` is removed, so progress is always locked in.

## Results

The same 45 cases, run on the code before this work (commit `7c9dcee`) and after it. The "before" column
was produced by exporting that commit and running the final case file against it, so both columns
measure the same thing.

| Measure | Before | After |
| --- | --- | --- |
| Cases passing | 26 / 45 | 45 / 45 |
| Expected identities found | 25 / 38 | 38 / 38 |
| Spurious (forbidden) identities shown | 8 | 0 |
| Mean applicable identities per input | 2.36 | 2.09 |
| Rules in the handbook | 39 | 69 (144 entries) |

Latency median was 62 ms after and 120 ms before on this machine; timings are noisy (the first call in a
process is slow), so the only claim is that nothing got slower in a way the run can see.

What changed, in the order it was measured:

1. **Benchmark first** (no behaviour change): 26/26 on the cases that existed then, with 13 expected
   identities missing and 8 spurious matches recorded as pending.
2. **Matcher fix**: spurious matches 8 to 0, mean applicable matches 2.33 to 1.42 (on that case set). A rule
   written as `n*w` (like `sin(2*a)`) no longer matches by binding `w` to a fraction of the user's term, so
   `sin(x)`, `sin(3x)`, `sin(x/2)` and `sin(x+y)` stop reporting the double-angle rule while `sin(2x)`,
   `sin(6x)` and `sin(2x+2y)` still match. Inputs such as `sin(2x/3)` are conservatively not offered as
   double angles.
3. **Retrieval fix** found by the benchmark: integer powers of 4 and above produced no index features,
   so `a**4 - b**4` could never be retrieved. They now map to `op:power`.
4. **30 generated entries** (`generate.py`): the 13 missing identities are found. The mean applicable
   count rises again because more true identities now apply, which is the point, not a regression.

## Limits of this evidence

- 45 cases, written by the same agent that wrote the generator; a maintainer adding cases the agent
  would not think of is the best next improvement.
- It measures the identity finder, not the chat agent, and not real user questions.
- "Found" means the right entry is applicable and verified for that input, not that the user wanted it.
- Four existing rules cannot fire on their own left-hand side by design (see `NOT_SELF_APPLICABLE` in
  `packages/knowledge/tests/test_benchmark.py`): SymPy evaluates `log(exp(a))` and `sqrt(a**2)` itself
  once the assumption is declared, and `hyp.cosh_cos` needs the imaginary unit, which the matcher discards.

## Adding content

Add benchmark cases first, as `pending` if the entry does not exist, then add the entry (a seed in
`generate.py` or hand-written YAML) and remove `pending` when the suite tells you to.
