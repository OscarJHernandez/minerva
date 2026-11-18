# Agent benchmark: false-derivative-recheck

Generated 2026-11-18 by `scripts/eval_models.py`. Case set `59f5a5fae33b` (1 cases, 3 repeat(s), seed 7, temperature 0.2). Code `1a1cf48dd2-dirty`, prompt `3125a4c9ab20`. The targets are goals, not results.

Model parameters: num_ctx 32768, num_predict None, temperature 0.2, top_p None, top_k None, think False, history_messages 20, history_tokens None, toolset auto.
Deterministic routing: on.

## Model under test

| Model | Weights as the host reports them | Defaults the host applies | Window loaded | Minerva sends | Thinking requested | Same weights through the run |
| --- | --- | --- | --- | --- | --- | --- |
| minerva | based on gemma4:26b, gemma4, 25.2B, Q4_K_M, digest 149a0ff8a1ca | top_k 64, top_p 0.9, draft_num_predict 3, num_ctx 16384, num_predict 1200, temperature 0.2 | 32768 | num_ctx 32768, num_predict None, temperature 0.2, top_p None, top_k None (num_predict is left to the host: 1200) | off (model can think) | yes |

Seed 7, 3 repeat(s), compute timeout 10.0 s, request timeout 180.0 s; prompt `3125a4c9ab20`, code `1a1cf48dd2-dirty`; Ollama 0.40.3.

## Headline

| Model | Passed | Tool selection | Valid arguments | Completion | Unsupported shown as verified | Unnecessary calls | Median s | p95 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| minerva | 3/3 (100.0%) | 0.0% | 100.0% | 100.0% | 0 | 3 of 6 | 6.2 | 9.4 |

Targets: tool selection >= 95%, valid arguments >= 99%, completion >= 90%, unsupported formulas shown as verified = 0.

## minerva

| Target | Value | Status |
| --- | --- | --- |
| tool selection accuracy | 0.0% | NOT met |
| valid arguments rate | 100.0% | met |
| completion rate | 100.0% | met |
| false verified | 0 | met |

### By category

| Category | Passed | Tool selection |
| --- | --- | --- |
| false_claim | 3/3 (100.0%) | 0.0% |

### Tool selection (expected rows, first tool called columns)

| expected \ called | check_identity |
| --- | --- |
| differentiate | 3 |

### By expected tool

| Tool | Passed | Tool selection | Invalid calls |
| --- | --- | --- | --- |
| differentiate | 3/3 | 0.0% | 0 of 6 |

### Latency

| | median | p95 | max |
| --- | --- | --- | --- |
| whole turn (s) | 6.2 | 9.4 | 9.4 |
| model calls (ms) | 3521 | 6736 | 6736 |
| tool calls (ms) | 1790 | 1820 | 1820 |

Tokens in/out: 58518 / 422.

First output after 406 ms (median) · tool calls that errored: 0 · tool results not verified: 3 · provider rate limits / timeouts: 0 / 0 · cost: n/a.
Context: window 32768 tokens, largest prompt 6623.
Verified-result coverage: 100.0% of 3 runs that expect a card · model calls per run: median 3.0, mean 3.0 · multi-turn consistency: n/a of 0 runs.

### Failures by class

No failures.

Formulas the model wrote: 6 unchecked, 0 contradicted, 0 correct ones marked contradicted. Repeated identical calls: 0.

