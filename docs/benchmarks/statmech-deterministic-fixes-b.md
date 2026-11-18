# Agent benchmark: statmech-deterministic-fixes-b

Generated 2026-11-18 by `scripts/eval_models.py`. Case set `41ac21a20cab` (9 cases, 3 repeat(s), seed 7, temperature 0.2). Code `a480474262`, prompt `ae2c8e30ab6a`. The targets are goals, not results.

Model parameters: num_ctx 32768, num_predict None, temperature 0.2, top_p None, top_k None, think False, history_messages 20, history_tokens None, toolset auto.
Deterministic routing: on.

## Model under test

| Model | Weights as the host reports them | Defaults the host applies | Window loaded | Minerva sends | Thinking requested | Same weights through the run |
| --- | --- | --- | --- | --- | --- | --- |
| minerva | based on gemma4:26b, gemma4, 25.2B, Q4_K_M, digest 149a0ff8a1ca | top_k 64, top_p 0.9, draft_num_predict 3, num_ctx 16384, num_predict 1200, temperature 0.2 | 32768 | num_ctx 32768, num_predict None, temperature 0.2, top_p None, top_k None (num_predict is left to the host: 1200) | off (model can think) | yes |

Seed 7, 3 repeat(s), compute timeout 10.0 s, request timeout 180.0 s; prompt `ae2c8e30ab6a`, code `a480474262`; Ollama 0.40.3.

## Headline

| Model | Passed | Tool selection | Valid arguments | Completion | Unsupported shown as verified | Unnecessary calls | Median s | p95 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| minerva | 10/27 (37.0%) | 63.0% | 72.3% | 77.8% | 0 | 62 of 83 | 16.8 | 28.3 |

Targets: tool selection >= 95%, valid arguments >= 99%, completion >= 90%, unsupported formulas shown as verified = 0.

## minerva

| Target | Value | Status |
| --- | --- | --- |
| tool selection accuracy | 63.0% | NOT met |
| valid arguments rate | 72.3% | NOT met |
| completion rate | 77.8% | NOT met |
| false verified | 0 | met |

### By category

| Category | Passed | Tool selection |
| --- | --- | --- |
| physics | 7/18 (38.9%) | 77.8% |
| unsupported | 3/3 (100.0%) | 100.0% |
| false_claim | 0/6 (0.0%) | 0.0% |

### Tool selection (expected rows, first tool called columns)

| expected \ called | matrix_calculus | none | statistical_mechanics |
| --- | --- | --- | --- |
| statistical_mechanics | 3 | 7 | 17 |

### By expected tool

| Tool | Passed | Tool selection | Invalid calls |
| --- | --- | --- | --- |
| statistical_mechanics | 10/27 | 63.0% | 23 of 83 |

### Latency

| | median | p95 | max |
| --- | --- | --- | --- |
| whole turn (s) | 16.8 | 28.3 | 28.3 |
| model calls (ms) | 14870 | 26629 | 26649 |
| tool calls (ms) | 882 | 3915 | 4784 |

Tokens in/out: 1137012 / 8877.

First output after 538 ms (median) · tool calls that errored: 27 · tool results not verified: 25 · provider rate limits / timeouts: 0 / 0 · cost: n/a.
Context: window 32768 tokens, largest prompt 13168.
Verified-result coverage: 75.0% of 16 runs that expect a card · model calls per run: median 5.0, mean 4.1 · multi-turn consistency: n/a of 0 runs.

### Failures by class

| Class | Runs |
| --- | --- |
| tool_selection | 7 |
| arguments | 4 |
| unproductive_loop | 3 |
| resource_limit | 3 |

Formulas the model wrote: 83 unchecked, 0 contradicted, 0 correct ones marked contradicted. Repeated identical calls: 6.

Unstable across repeats (0.47 std of passes per repeat): `statmech-oscillator-heat-capacity`, `statmech-stirling-is-an-approximation`.

### Failed runs

- `statmech-ideal-gas-conventions-are-the-users` (tool_selection): expected one of ['statistical_mechanics'], called none
- `statmech-wrong-partition-function-is-not-confirmed` (tool_selection): expected one of ['statistical_mechanics'], called none
- `statmech-noncommuting-response-is-not-claimed` (unproductive_loop): the turn ended early: repeated_calls; expected one of ['statistical_mechanics'], called ['matrix_calculus']; reply does not say what it should
- `statmech-fermi-number-and-variance` (arguments): the tool's result is not the expected one (different input or status)
- `statmech-ising-thermodynamic-limit` (resource_limit): the turn ended early: step_limit
- `statmech-stirling-is-an-approximation` (tool_selection): expected one of ['statistical_mechanics'], called none
- `statmech-oscillator-heat-capacity` (arguments): every call to the tool had invalid arguments

