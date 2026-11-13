# Distribution-calculus benchmark

Reproducible measurement of Minerva's Dirac delta and Heaviside calculus (`distribution` operation).
The cases are problems from mathematical physics with **hand-written, analytic answers**; the engine
is never its own oracle. Cases live in
`packages/knowledge/src/minerva_knowledge/data/benchmark/distribution_cases.yaml`.

```bash
uv run python -m minerva_knowledge benchmark --only distributions                  # run and print
uv run python -m minerva_knowledge benchmark --only distributions --json out.json  # machine-readable
uv run python -m minerva_knowledge benchmark --only distributions --compare docs/benchmarks/distributions.json
```

`--compare` prints each metric next to the committed baseline and names every case that regressed,
was fixed or changed result; it exits 1 on any regression. The plain `benchmark` command runs this
part too. Latency is reported, never asserted.

## Case kinds

| Kind | Meaning | Passes when |
| --- | --- | --- |
| `supported` | a problem the first version solves | status `ok`, result equal to the analytic one, every listed handbook identity cited, none of the tempting wrong answers returned, verification passed |
| `unsupported` | outside the first version, or conditions that cannot be established | refused with the expected status and reason, and no result shown |
| `invalid_transform` | a mathematically wrong rewriting (missing 1/\|a\|, wrong sign, dropped root, wrong jump, wrong window, wrong endpoint convention) | rejected by the test-function pairing |

## Baseline (v1)

Stored in [`benchmarks/distributions.json`](benchmarks/distributions.json).

| Metric | Value |
| --- | --- |
| Cases passed | 54 / 54 |
| Supported cases correct | 29 / 29 |
| Refused as designed | 15 / 15 (5 `assumptions_insufficient`, 1 `integration_domain_ambiguous`, 1 `singular_root`, 8 `unsupported_distribution`) |
| Invalid rewritings rejected | 10 / 10 |
| Coverage (supported / supported + refused) | 66 % |
| Accepted results with passed verification | 100 % |
| Handbook retrieval recall@1 / @3 / @5 | 7 % / 86 % / 100 % (MRR 0.38) |
| Latency median / p95 | about 0.1 s / about 1.5 s (a parametric case runs three parameter samples) |

Retrieval is the weak point and is reported as measured: the seven delta entries all share the
features `f:diracdelta` and the action, so the right one is usually found, but rarely first. A
retriever that reads the structure of the argument (linear or polynomial, with or without a
multiplier) would raise recall@1; the benchmark will show it.

## How to read a failure

- `expected ok, got <status> (<reason>)` on a supported case: a coverage regression.
- `result … != …` or `returned the forbidden result`: a correctness regression. Treat as a bug in the
  engine, not in the case.
- `the invalid rewriting was accepted`: the verifier lost power. This is the most serious failure.
- `a result was shown` on an unsupported case: the engine answered where it should have refused.

## Limits of the evidence

The pairing check (Gaussian/erf regularisation of the input, three halving widths, Richardson
extrapolation, six Gaussian-windowed test functions, up to three parameter samples) is supporting
evidence, not a proof. The conditions leg (real variable, nonzero scale, simple real roots, positions
decided) is exact and reported separately. See `docs/architecture.md` for the semantics.
