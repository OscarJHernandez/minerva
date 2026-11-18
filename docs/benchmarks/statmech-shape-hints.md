# Agent benchmark: statmech-shape-hints

Generated 2026-11-18 by `scripts/eval_models.py`. Case set `41ac21a20cab` (9 cases, 3 repeat(s), seed 7, temperature 0.2). Code `3771e65d67`, prompt `1f6f9104c4ba`. The targets are goals, not results.

Model parameters: num_ctx 32768, num_predict None, temperature 0.2, top_p None, top_k None, think False, history_messages 20, history_tokens None, toolset auto.
Deterministic routing: on.

## Model under test

| Model | Weights as the host reports them | Defaults the host applies | Window loaded | Minerva sends | Thinking requested | Same weights through the run |
| --- | --- | --- | --- | --- | --- | --- |
| minerva | based on gemma4:26b, gemma4, 25.2B, Q4_K_M, digest 149a0ff8a1ca | top_k 64, top_p 0.9, draft_num_predict 3, num_ctx 16384, num_predict 1200, temperature 0.2 | 32768 | num_ctx 32768, num_predict None, temperature 0.2, top_p None, top_k None (num_predict is left to the host: 1200) | off (model can think) | yes |

Seed 7, 3 repeat(s), compute timeout 10.0 s, request timeout 180.0 s; prompt `1f6f9104c4ba`, code `3771e65d67`; Ollama 0.40.3.

## Headline

| Model | Passed | Tool selection | Valid arguments | Completion | Unsupported shown as verified | Unnecessary calls | Median s | p95 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| minerva | 6/27 (22.2%) | 59.3% | 50.7% | 100.0% | 0 | 50 of 73 | 17.7 | 22.3 |

Targets: tool selection >= 95%, valid arguments >= 99%, completion >= 90%, unsupported formulas shown as verified = 0.

## minerva

| Target | Value | Status |
| --- | --- | --- |
| tool selection accuracy | 59.3% | NOT met |
| valid arguments rate | 50.7% | NOT met |
| completion rate | 100.0% | met |
| false verified | 0 | met |

### By category

| Category | Passed | Tool selection |
| --- | --- | --- |
| physics | 3/18 (16.7%) | 72.2% |
| unsupported | 3/3 (100.0%) | 100.0% |
| false_claim | 0/6 (0.0%) | 0.0% |

### Tool selection (expected rows, first tool called columns)

| expected \ called | matrix_calculus | none | statistical_mechanics |
| --- | --- | --- | --- |
| statistical_mechanics | 3 | 8 | 16 |

### By expected tool

| Tool | Passed | Tool selection | Invalid calls |
| --- | --- | --- | --- |
| statistical_mechanics | 6/27 | 59.3% | 36 of 73 |

### Latency

| | median | p95 | max |
| --- | --- | --- | --- |
| whole turn (s) | 17.7 | 22.3 | 25.0 |
| model calls (ms) | 16795 | 21321 | 24056 |
| tool calls (ms) | 30 | 2077 | 2119 |

Tokens in/out: 1090075 / 10516.

First output after 530 ms (median) · tool calls that errored: 42 · tool results not verified: 7 · provider rate limits / timeouts: 0 / 0 · cost: n/a.
Context: window 32768 tokens, largest prompt 12757.
Verified-result coverage: 60.0% of 10 runs that expect a card · model calls per run: median 4.0, mean 3.9 · multi-turn consistency: n/a of 0 runs.

### Failures by class

| Class | Runs |
| --- | --- |
| tool_selection | 11 |
| arguments | 7 |
| parsing | 3 |

Formulas the model wrote: 124 unchecked, 1 contradicted, 0 correct ones marked contradicted. Repeated identical calls: 4.

### Failed runs

- `statmech-oscillator-heat-capacity` (arguments): every call to the tool had invalid arguments
- `statmech-ideal-gas-conventions-are-the-users` (tool_selection): expected one of ['statistical_mechanics'], called none
- `statmech-wrong-partition-function-is-not-confirmed` (tool_selection): expected one of ['statistical_mechanics'], called none
- `statmech-noncommuting-response-is-not-claimed` (tool_selection): expected one of ['statistical_mechanics'], called ['matrix_calculus']; reply does not say what it should
- `statmech-fermi-number-and-variance` (arguments): the tool's result is not the expected one (different input or status)
- `statmech-ising-thermodynamic-limit` (parsing): the tool's result is not the expected one (different input or status)
- `statmech-stirling-is-an-approximation` (arguments): every call to the tool had invalid arguments

