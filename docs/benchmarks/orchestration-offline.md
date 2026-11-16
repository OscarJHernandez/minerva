# Offline orchestration suite

The real agent loop, tools and claim check with a replayed model (no network). Tokens are estimates (`budget.estimate_tokens`, deliberately pessimistic); timings are one run on one machine.

16/17 trajectories end mathematically correct · 40 model calls · 24 tool calls (2 invalid, 1 unknown names) · ≈479404 tokens in, ≈1462 out · 10 unbacked claims

| Trajectory | Category | Model calls | Tool calls (ok) | Routed | Tokens in | Evidence | Math ok | Tool ms | Claim ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wedge-original-failure | notation | 4 | 3 (0) | 0 | 57164 | 0/4 | NO | 2514 | 527 |
| pendulum-original-failure | notation | 4 | 3 (2) | 0 | 59866 | 4/4 | yes | 1776 | 982 |
| pendulum-routed-latex | lagrangian | 1 | 1 (1) | 1 | 14923 | 4/4 | yes | 0 | 907 |
| pendulum-routed-text-velocities | notation | 1 | 1 (1) | 1 | 14913 | 4/4 | yes | 0 | 944 |
| pendulum-model-calls-with-text | notation | 2 | 1 (1) | 0 | 28990 | 4/4 | yes | 888 | 978 |
| pendulum-assignment-and-greek | notation | 2 | 1 (1) | 0 | 28991 | 4/4 | yes | 900 | 955 |
| pendulum-wrong-sign-stays-unbacked | lagrangian | 2 | 1 (1) | 0 | 29004 | 3/4 | yes | 888 | 1461 |
| cyclic-oscillator | conserved | 2 | 1 (1) | 0 | 28672 | 1/1 | yes | 489 | 464 |
| simplify-with-functions-prefix | algebra | 2 | 1 (1) | 0 | 17214 | 1/2 | yes | 1010 | 399 |
| simplify-text-syntax-then-fix | notation | 3 | 2 (1) | 0 | 25914 | 1/1 | yes | 2249 | - |
| unreadable-arguments-then-valid | recovery | 3 | 2 (1) | 0 | 25791 | 1/2 | yes | 985 | 403 |
| tool-call-written-as-prose | recovery | 3 | 1 (1) | 0 | 25698 | 1/1 | yes | 611 | - |
| unknown-tool-is-refused | recovery | 3 | 2 (1) | 0 | 26042 | 1/1 | yes | 585 | - |
| integrate-antiderivative | calculus | 2 | 1 (1) | 0 | 17101 | 1/1 | yes | 589 | - |
| matrix-expansion-noncommuting | matrix | 2 | 1 (1) | 0 | 21923 | 0/2 | yes | 394 | 386 |
| ode-first-order | ode | 2 | 1 (1) | 0 | 27037 | 0/1 | yes | 459 | 561 |
| vector-divergence-product | vector | 2 | 1 (1) | 0 | 30161 | 0/1 | yes | 597 | 377 |
