# Offline orchestration suite

The real agent loop, tools and claim check with a replayed model (no network). Tokens are estimates (`budget.estimate_tokens`, deliberately pessimistic); timings are one run on one machine.

26/26 trajectories end mathematically correct · 60 model calls · 36 tool calls (2 invalid, 1 unknown names) · ≈724313 tokens in, ≈2202 out · 7 unbacked claims

| Trajectory | Category | Model calls | Tool calls (ok) | Routed | Without a model call | Tokens in | Evidence | Math ok | Tool ms | Claim ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wedge-original-failure | notation | 4 | 3 (1) | 0 | 2 refused | 49463 | 4/4 | yes | 2419 | 368 |
| wedge-xdot-first-call | notation | 2 | 1 (1) | 0 | - | 29217 | 4/4 | yes | 809 | 370 |
| wedge-wrong-format-recovered | recovery | 2 | 1 (1) | 0 | 1 re-read | 29244 | 4/4 | yes | 1162 | 368 |
| wedge-shape-errors-repaired | recovery | 2 | 1 (1) | 0 | 1 adjusted | 29287 | 4/4 | yes | 792 | 372 |
| ambiguous-spelling-is-refused-then-fixed | notation | 3 | 2 (1) | 0 | - | 43158 | - | yes | 822 | - |
| equivalent-failed-call-is-not-rerun | recovery | 3 | 2 (0) | 0 | 1 refused | 25805 | - | yes | 721 | - |
| unsupported-is-not-retried | recovery | 3 | 2 (0) | 0 | 1 refused | 25842 | - | yes | 382 | - |
| wedge-routed-xdot | notation | 1 | 1 (1) | 1 | - | 15075 | 4/4 | yes | 820 | 372 |
| pendulum-original-failure | notation | 4 | 3 (2) | 0 | 1 refused | 59959 | 4/4 | yes | 1650 | 884 |
| pendulum-routed-latex | lagrangian | 1 | 1 (1) | 1 | - | 14947 | 4/4 | yes | 821 | 883 |
| pendulum-routed-text-velocities | notation | 1 | 1 (1) | 1 | - | 14937 | 4/4 | yes | 820 | 877 |
| pendulum-model-calls-with-text | notation | 2 | 1 (1) | 0 | - | 29038 | 4/4 | yes | 812 | 873 |
| pendulum-assignment-and-greek | notation | 2 | 1 (1) | 0 | 1 adjusted | 29065 | 4/4 | yes | 816 | 881 |
| pendulum-wrong-sign-stays-unbacked | lagrangian | 2 | 1 (1) | 0 | - | 29051 | 3/4 | yes | 858 | 1399 |
| cyclic-oscillator | conserved | 2 | 1 (1) | 0 | - | 28719 | 1/1 | yes | 433 | 433 |
| simplify-with-functions-prefix | algebra | 2 | 1 (1) | 0 | - | 17261 | 1/2 | yes | 919 | 380 |
| simplify-text-syntax-then-fix | notation | 3 | 2 (2) | 0 | 1 re-read | 26322 | 1/1 | yes | 4142 | - |
| simplify-text-syntax-recovered | recovery | 2 | 1 (1) | 0 | 1 re-read | 17277 | 1/1 | yes | 2451 | - |
| unreadable-arguments-then-valid | recovery | 3 | 2 (1) | 0 | - | 25861 | 1/2 | yes | 934 | 364 |
| tool-call-written-as-prose | recovery | 3 | 1 (1) | 0 | - | 25767 | 1/1 | yes | 561 | - |
| unknown-tool-is-refused | recovery | 3 | 2 (1) | 0 | - | 26112 | 1/1 | yes | 535 | - |
| integrate-antiderivative | calculus | 2 | 1 (1) | 0 | - | 17148 | 1/1 | yes | 536 | - |
| matrix-expansion-noncommuting | matrix | 2 | 1 (1) | 0 | - | 21970 | 0/2 | yes | 389 | 365 |
| ode-first-order | ode | 2 | 1 (1) | 0 | - | 27084 | 0/1 | yes | 435 | 508 |
| vector-divergence-product | vector | 2 | 1 (1) | 0 | - | 31856 | 0/1 | yes | 534 | 363 |
| vector-distributional-divergence | vector | 2 | 1 (1) | 0 | - | 34848 | 1/2 | yes | 661 | 363 |
