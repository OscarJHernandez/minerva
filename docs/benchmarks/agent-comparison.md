# Baseline against new

Baseline `baseline` (code `baseline-7b19980`, cases `e89efb8efe53`) against `new` (code `abaeb9fd4b`, cases `e89efb8efe53`).

## gemma4:26b

| Metric | Baseline | New | Change |
| --- | --- | --- | --- |
| pass rate | 51.3% | 48.1% | -3.2 points |
| tool selection | 75.3% | 77.3% | +1.9 points |
| valid arguments | 84.9% | 91.9% | +7.0 points |
| completion | 98.7% | 94.8% | -3.9 points |
| unsupported shown as verified | 4 | 4 | +0 |
| unnecessary calls | 34 | 27 | -7 |
| repeated calls | 15 | 10 | -5 |
| checker false positives | 0 | 0 | +0 |
| turn latency (s) median | 23.8 | 26.7 | |
| turn latency (s) p95 | 53.7 | 55.6 | |
| model call (ms) median | 23118 | 25969 | |
| model call (ms) p95 | 51570 | 55305 | |
| tool call (ms) median | 476 | 474 | |
| tool call (ms) p95 | 1930 | 1927 | |

### By category

| Category | Baseline | New |
| --- | --- | --- |
| symbolic | 77.8% | 77.8% |
| multistep | 75.0% | 75.0% |
| calculus | 18.2% | 18.2% |
| retrieval | 77.8% | 77.8% |
| tensor_matrix | 10.0% | 20.0% |
| physics | 77.8% | 61.1% |
| ambiguous | 66.7% | 66.7% |
| unsupported | 0.0% | 0.0% |
| false_claim | 33.3% | 27.8% |
| followup | 60.0% | 40.0% |

### Failures by class

| Class | Baseline | New |
| --- | --- | --- |
| arguments | 11 | 12 |
| model_honesty | 22 | 23 |
| resource_limit | 2 | 0 |
| tool_selection | 36 | 33 |
| unproductive_loop | 0 | 8 |
| verification | 4 | 4 |

Cases that improved (3): `false-derivative`, `perturbation-duffing`, `tensor-riemann-pair-symmetry`

Cases that got worse (5): `assumptions-carried`, `false-energy-units`, `followup-keeps-assumptions`, `motion-values-given`, `units-check`

Differences of one case between single runs are within the run-to-run noise of a sampling model; use repeats before reading a change into them.

## qwen3.5:35b

| Metric | Baseline | New | Change |
| --- | --- | --- | --- |
| pass rate | 66.2% | 66.2% | +0.0 points |
| tool selection | 89.0% | 85.1% | -3.9 points |
| valid arguments | 99.3% | 99.3% | +0.0 points |
| completion | 99.4% | 99.4% | +0.0 points |
| unsupported shown as verified | 3 | 3 | +0 |
| unnecessary calls | 16 | 21 | +5 |
| repeated calls | 0 | 0 | +0 |
| checker false positives | 0 | 0 | +0 |
| turn latency (s) median | 27.1 | 26.7 | |
| turn latency (s) p95 | 62.3 | 72.1 | |
| model call (ms) median | 26137 | 25458 | |
| model call (ms) p95 | 61434 | 71077 | |
| tool call (ms) median | 402 | 502 | |
| tool call (ms) p95 | 1657 | 1706 | |

### By category

| Category | Baseline | New |
| --- | --- | --- |
| symbolic | 77.8% | 88.9% |
| multistep | 87.5% | 87.5% |
| calculus | 36.4% | 45.5% |
| retrieval | 88.9% | 88.9% |
| tensor_matrix | 50.0% | 20.0% |
| physics | 94.4% | 94.4% |
| ambiguous | 91.7% | 100.0% |
| unsupported | 41.7% | 25.0% |
| false_claim | 27.8% | 33.3% |
| followup | 70.0% | 60.0% |

### Failures by class

| Class | Baseline | New |
| --- | --- | --- |
| arguments | 6 | 3 |
| missing_assumptions | 2 | 2 |
| model_honesty | 24 | 21 |
| resource_limit | 1 | 1 |
| tool_selection | 16 | 22 |
| verification | 3 | 3 |

Cases that improved (6): `ask-when-ambiguous`, `false-log-sum`, `integrate-gaussian`, `integrate-improper`, `sqrt-no-assumptions`, `unsupported-no-closed-form`

Cases that got worse (7): `false-energy-units`, `followup-uses-context`, `matrix-commuting-declared`, `tensor-contraction`, `unsupported-derivative-abs`, `unsupported-series-pole`, `unsupported-transcendental`

Differences of one case between single runs are within the run-to-run noise of a sampling model; use repeats before reading a change into them.
