# Offline orchestration suite

The real agent loop, tools and claim check with a replayed model (no network). Tokens are estimates (`budget.estimate_tokens`, deliberately pessimistic); timings are one run on one machine.

26/26 trajectories end mathematically correct · 60 model calls · 36 tool calls (2 invalid, 1 unknown names) · ≈727858 tokens in, ≈2202 out · 5 unbacked claims

| Trajectory | Category | Model calls | Tool calls (ok) | Routed | Without a model call | Tokens in | Evidence | Math ok | Tool ms | Claim ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wedge-original-failure | notation | 4 | 3 (1) | 0 | 2 refused | 49931 | 4/4 | yes | 4008 | 608 |
| wedge-xdot-first-call | notation | 2 | 1 (1) | 0 | - | 29456 | 4/4 | yes | 881 | 467 |
| wedge-wrong-format-recovered | recovery | 2 | 1 (1) | 0 | 1 re-read | 29484 | 4/4 | yes | 1777 | 416 |
| wedge-shape-errors-repaired | recovery | 2 | 1 (1) | 0 | 1 adjusted | 29526 | 4/4 | yes | 916 | 425 |
| ambiguous-spelling-is-refused-then-fixed | notation | 3 | 2 (1) | 0 | - | 43517 | - | yes | 934 | - |
| equivalent-failed-call-is-not-rerun | recovery | 3 | 2 (0) | 0 | 1 refused | 25877 | - | yes | 834 | - |
| unsupported-is-not-retried | recovery | 3 | 2 (0) | 0 | 1 refused | 25913 | - | yes | 424 | - |
| wedge-routed-xdot | notation | 1 | 1 (1) | 1 | - | 15195 | 4/4 | yes | 950 | 425 |
| pendulum-original-failure | notation | 4 | 3 (2) | 0 | 1 refused | 60438 | 4/4 | yes | 1883 | 1234 |
| pendulum-routed-latex | lagrangian | 1 | 1 (1) | 1 | - | 15066 | 4/4 | yes | 991 | 1282 |
| pendulum-routed-text-velocities | notation | 1 | 1 (1) | 1 | - | 15056 | 4/4 | yes | 1282 | 1265 |
| pendulum-model-calls-with-text | notation | 2 | 1 (1) | 0 | - | 29276 | 4/4 | yes | 1283 | 1188 |
| pendulum-assignment-and-greek | notation | 2 | 1 (1) | 0 | 1 adjusted | 29303 | 4/4 | yes | 1219 | 1141 |
| pendulum-wrong-sign-stays-unbacked | lagrangian | 2 | 1 (1) | 0 | - | 29290 | 3/4 | yes | 1236 | 1882 |
| cyclic-oscillator | conserved | 2 | 1 (1) | 0 | - | 28958 | 1/1 | yes | 778 | 754 |
| simplify-with-functions-prefix | algebra | 2 | 1 (1) | 0 | - | 17305 | 2/2 | yes | 1576 | - |
| simplify-text-syntax-then-fix | notation | 3 | 2 (2) | 0 | 1 re-read | 26388 | 1/1 | yes | 5837 | - |
| simplify-text-syntax-recovered | recovery | 2 | 1 (1) | 0 | 1 re-read | 17321 | 1/1 | yes | 3291 | - |
| unreadable-arguments-then-valid | recovery | 3 | 2 (1) | 0 | - | 25928 | 2/2 | yes | 1368 | - |
| tool-call-written-as-prose | recovery | 3 | 1 (1) | 0 | - | 25833 | 1/1 | yes | 858 | - |
| unknown-tool-is-refused | recovery | 3 | 2 (1) | 0 | - | 26187 | 1/1 | yes | 783 | - |
| integrate-antiderivative | calculus | 2 | 1 (1) | 0 | - | 17192 | 1/1 | yes | 634 | - |
| matrix-expansion-noncommuting | matrix | 2 | 1 (1) | 0 | - | 22014 | 0/2 | yes | 718 | 472 |
| ode-first-order | ode | 2 | 1 (1) | 0 | - | 27128 | 0/1 | yes | 747 | 589 |
| vector-divergence-product | vector | 2 | 1 (1) | 0 | - | 31642 | 0/1 | yes | 917 | 485 |
| vector-distributional-divergence | vector | 2 | 1 (1) | 0 | - | 34634 | 1/2 | yes | 930 | 422 |
