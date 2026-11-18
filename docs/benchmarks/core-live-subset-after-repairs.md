# Agent benchmark: core-subset-after-repairs

Generated 2026-11-18 by `scripts/eval_models.py`. Case set `14c989879f3c` (29 cases, 1 repeat(s), seed 7, temperature 0.2). Code `e5f9a5011e-dirty`, prompt `1f6f9104c4ba`. The targets are goals, not results.

Model parameters: num_ctx 32768, num_predict None, temperature 0.2, top_p None, top_k None, think False, history_messages 20, history_tokens None, toolset auto.
Deterministic routing: on.

## Headline

| Model | Passed | Tool selection | Valid arguments | Completion | Unsupported shown as verified | Unnecessary calls | Median s | p95 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| minerva | 10/29 (34.5%) | 82.8% | 83.6% | 82.8% | 0 | 27 of 61 | 11.1 | 20.2 |

Targets: tool selection >= 95%, valid arguments >= 99%, completion >= 90%, unsupported formulas shown as verified = 0.

## minerva

| Target | Value | Status |
| --- | --- | --- |
| tool selection accuracy | 82.8% | NOT met |
| valid arguments rate | 83.6% | NOT met |
| completion rate | 82.8% | NOT met |
| false verified | 0 | met |

### By category

| Category | Passed | Tool selection |
| --- | --- | --- |
| calculus | 5/11 (45.5%) | 90.9% |
| physics | 2/9 (22.2%) | 77.8% |
| ambiguous | 1/5 (20.0%) | 80.0% |
| unsupported | 2/2 (100.0%) | 100.0% |
| false_claim | 0/2 (0.0%) | 50.0% |

### Tool selection (expected rows, first tool called columns)

| expected \ called | creative_telescoping | distribution | electromagnetism | none | search_handbook | vector_calculus |
| --- | --- | --- | --- | --- | --- | --- |
| creative_telescoping | 3 |  |  |  |  |  |
| distribution |  | 4 |  |  |  |  |
| electromagnetism |  |  | 3 | 1 | 1 |  |
| vector_calculus |  |  | 1 | 2 |  | 14 |

### By expected tool

| Tool | Passed | Tool selection | Invalid calls |
| --- | --- | --- | --- |
| creative_telescoping | 1/3 | 100.0% | 1 of 6 |
| distribution | 2/4 | 100.0% | 5 of 7 |
| electromagnetism | 1/5 | 60.0% | 1 of 11 |
| vector_calculus | 6/17 | 82.4% | 3 of 37 |

### Latency

| | median | p95 | max |
| --- | --- | --- | --- |
| whole turn (s) | 11.1 | 20.2 | 21.3 |
| model calls (ms) | 10730 | 19388 | 19719 |
| tool calls (ms) | 753 | 1592 | 2091 |

Tokens in/out: 981454 / 8818.

First output after 845 ms (median) · tool calls that errored: 8 · tool results not verified: 24 · provider rate limits / timeouts: 0 / 0 · cost: n/a.
Context: window 32768 tokens, largest prompt 15638.
Verified-result coverage: 66.7% of 24 runs that expect a card · model calls per run: median 3.0, mean 3.3 · multi-turn consistency: n/a of 0 runs.

### Failures by class

| Class | Runs |
| --- | --- |
| arguments | 7 |
| unproductive_loop | 4 |
| tool_selection | 4 |
| parsing | 2 |
| model_honesty | 1 |
| model_host | 1 |

Formulas the model wrote: 123 unchecked, 0 contradicted, 0 correct ones marked contradicted. Repeated identical calls: 3.

### Failed runs

- `delta-integral-uses-the-tool` (unproductive_loop): the turn ended early: repeated_calls; the tool's result is not the expected one (different input or status)
- `delta-on-a-limit-asks-for-a-convention` (arguments): every call to the tool had invalid arguments
- `vector-curl-grad-needs-smoothness` (arguments): the tool's result is not the expected one (different input or status)
- `vector-curl-grad-with-smoothness` (arguments): the tool's result is not the expected one (different input or status)
- `vector-curl-curl-rewrite` (arguments): arguments do not say what the user asked
- `vector-curl-curl-rewrite-needs-smoothness` (parsing): the tool's result is not the expected one (different input or status)
- `em-wave-equation-from-maxwell` (arguments): arguments do not say what the user asked
- `em-adopting-premises-is-the-users-choice` (tool_selection): expected one of ['electromagnetism'], called none
- `em-gauge-invariance` (unproductive_loop): the turn ended early: repeated_calls; arguments do not say what the user asked
- `es-laplace-rectangle` (tool_selection): expected one of ['electromagnetism'], called ['search_handbook']; reply does not say what it should
- `integral-flux-through-a-sphere` (tool_selection): expected one of ['vector_calculus'], called none; reply does not say what it should
- `integral-theorem-with-a-pole-is-refused` (tool_selection): expected one of ['vector_calculus'], called none; reply does not say what it should
- `telescoping-gaussian-parameter` (model_honesty): 7 unchecked formula(s) in the reply
- `telescoping-undeclared-sign-is-not-assumed` (parsing): the tool's result is not the expected one (different input or status)
- `distributional-translated-source` (unproductive_loop): the turn ended early: repeated_calls; the tool's result is not the expected one (different input or status); reply does not say what it should
- `distributional-point-charge-gauss` (unproductive_loop): the turn ended early: repeated_calls; the tool's result is not the expected one (different input or status); reply does not say what it should
- `distributional-classical-versus-distributional` (arguments): arguments do not say what the user asked; reply does not say what it should
- `distributional-wrong-normalisation-is-rejected` (arguments): arguments do not say what the user asked
- `homepage-coulomb-divergence` (model_host): the turn ended early: model_error; the tool's result is not the expected one (different input or status); reply does not say what it should

