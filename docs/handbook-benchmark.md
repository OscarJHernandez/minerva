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

## Retrieval quality

`benchmark` also reports where the expected identities rank among the candidates retrieval proposes (the
single-step cases above, 38 expected identities). Retrieval order is BM25 over the structural feature tokens, so
it is a coarse filter, not a ranking of usefulness; the engine decides what applies.

| Measure | Value |
| --- | --- |
| Expected identities in the candidate pool | 100% |
| Recall at 1 / 3 / 5 / 10 | 29% / 61% / 74% / 92% |
| Mean reciprocal rank | 0.497 |
| Mean candidate pool (single step / chain search) | 12.4 / 41.1 |
| Expected identities missing from the chain pool | 0 |

The pool, not the rank, is what matters today: the whole pool goes to the engine. Rank would start to matter
if the handbook grows enough that the pool must be capped (`test_default_candidate_pool_equals_uncapped_pool`
guards that). The tests set floors (pool 100%, recall at 10 of 90%, MRR 0.45), not targets.

## Multi-step transformations

`data/benchmark/transform_cases.yaml`: 32 hand-written cases run through `transform_expression`: chains of
two or more identities, identities inside nested expressions, an identity read right to left, the goals
`expand`, `factor` and `rewrite`, conditional identities that must stay blocked, and inputs where nothing
may be applied. A case states the expected status, the accepted result forms (structural equality, so an equal
but different form fails: this measures ranking, since correctness is the engine's verification), the identities
that must appear in the chain shown and those that must appear in no chain.

| Measure | Result |
| --- | --- |
| Cases passing | 32 / 32 (23 chains, 9 negative) |
| `goal: simplify` cases whose result is the expected form: chain search / SymPy `simplify` alone | 14 / 14 vs 13 / 14 |
| Identities cited across the cases | 27 |
| Latency per case (median / max, including the sandbox) | 266 ms / 2.8 s |

How to read this: SymPy's own simplification already finds the expected form in 13 of the 14 `goal: simplify`
cases, so the chain search mostly adds the cited, individually verified steps rather than a different answer. The
one case it adds is `\sin^2 x + \cos^2 x + \tan^2 x`, where SymPy returns `1/\cos^2 x` and the expected `\sec^2 x`
takes two identities; that is a choice of accepted form, so treat it as one data point, not a rate. The other gains do
not show in a "same form" count: the three goals other than `simplify` (SymPy `simplify` cannot be asked for
them), and the report of identities that match but are blocked by conditions. Latency is higher than the single
SymPy call (about 0.1 s) because every candidate is verified; the search stops after two levels without
improvement.

What the benchmark found and changed while it was written:

- A single term matched `sin^2 + cos^2` with a leftover term absorbed by the matcher's `rest` wildcard (true,
  but not what a chain of identities should be built from); chains use `strict` matching, `find_identities` keeps
  the lenient behaviour.
- Reading `sin a + sin b = 2 sin(a/2+b/2) cos(a/2-b/2)` backwards matched `2 sin x cos x` with `b = 0`: reverse
  matches that bind a variable to zero are refused, and the rule with fewer variables is cited first.
- With `expand` or `factor` a chain went through `exp` or `tan` to reach a product that was shorter but not
  what was asked for; those goals may not introduce functions other than the input's.

## Limits of this evidence

- 45 cases, written by the same agent that wrote the generator; a maintainer adding cases the agent
  would not think of is the best next improvement.
- It measures the identity finder and the chain search, not the chat agent, and not real user questions. The 32
  chain cases were written by the agent that wrote the search; a maintainer adding cases it would not think of
  is the best next improvement.
- The chain cases measure the form of the answer and the identities cited. SymPy's `simplify` is already strong on
  these inputs, so the case count shows no large gain in answers; the gain is in what is shown and checked.
- "Found" means the right entry is applicable and verified for that input, not that the user wanted it.
- Four existing rules cannot fire on their own left-hand side by design (see `NOT_SELF_APPLICABLE` in
  `packages/knowledge/tests/test_benchmark.py`): SymPy evaluates `log(exp(a))` and `sqrt(a**2)` itself
  once the assumption is declared, and `hyp.cosh_cos` needs the imaginary unit, which the matcher discards.

## Adding content

Add benchmark cases first, as `pending` if the entry does not exist, then add the entry (a seed in
`generate.py` or hand-written YAML) and remove `pending` when the suite tells you to.
