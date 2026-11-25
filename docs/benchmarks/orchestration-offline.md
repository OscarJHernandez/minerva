# Offline orchestration suite

The real agent loop, tools and claim check with a replayed model (no network). Tokens are estimates (`budget.estimate_tokens`, deliberately pessimistic); timings are one run on one machine.

26/26 trajectories end mathematically correct · 60 model calls · 36 tool calls (2 invalid, 1 unknown names) · ≈742967 tokens in, ≈2202 out · 5 unbacked claims

| Trajectory | Category | Model calls | Tool calls (ok) | Routed | Without a model call | Tokens in | Evidence | Math ok | Tool ms | Claim ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wedge-original-failure | notation | 4 | 3 (1) | 0 | 2 refused | 51376 | 4/4 | yes | 2962 | 475 |
| wedge-xdot-first-call | notation | 2 | 1 (1) | 0 | - | 30308 | 4/4 | yes | 989 | 479 |
| wedge-wrong-format-recovered | recovery | 2 | 1 (1) | 0 | 1 re-read | 30335 | 4/4 | yes | 1467 | 478 |
| wedge-shape-errors-repaired | recovery | 2 | 1 (1) | 0 | 1 adjusted | 30378 | 4/4 | yes | 959 | 474 |
| ambiguous-spelling-is-refused-then-fixed | notation | 3 | 2 (1) | 0 | - | 44771 | - | yes | 1036 | - |
| equivalent-failed-call-is-not-rerun | recovery | 3 | 2 (0) | 0 | 1 refused | 26110 | - | yes | 936 | - |
| unsupported-is-not-retried | recovery | 3 | 2 (0) | 0 | 1 refused | 26147 | - | yes | 479 | - |
| wedge-routed-xdot | notation | 1 | 1 (1) | 1 | - | 15643 | 4/4 | yes | 990 | 468 |
| pendulum-original-failure | notation | 4 | 3 (2) | 0 | 1 refused | 62226 | 4/4 | yes | 1998 | 1043 |
| pendulum-routed-latex | lagrangian | 1 | 1 (1) | 1 | - | 15514 | 4/4 | yes | 982 | 1051 |
| pendulum-routed-text-velocities | notation | 1 | 1 (1) | 1 | - | 15504 | 4/4 | yes | 987 | 1069 |
| pendulum-model-calls-with-text | notation | 2 | 1 (1) | 0 | - | 30128 | 4/4 | yes | 980 | 1054 |
| pendulum-assignment-and-greek | notation | 2 | 1 (1) | 0 | 1 adjusted | 30155 | 4/4 | yes | 977 | 1045 |
| pendulum-wrong-sign-stays-unbacked | lagrangian | 2 | 1 (1) | 0 | - | 30141 | 3/4 | yes | 985 | 1643 |
| cyclic-oscillator | conserved | 2 | 1 (1) | 0 | - | 29810 | 1/1 | yes | 544 | 534 |
| simplify-with-functions-prefix | algebra | 2 | 1 (1) | 0 | - | 17461 | 2/2 | yes | 1139 | - |
| simplify-text-syntax-then-fix | notation | 3 | 2 (2) | 0 | 1 re-read | 26622 | 1/1 | yes | 5064 | - |
| simplify-text-syntax-recovered | recovery | 2 | 1 (1) | 0 | 1 re-read | 17477 | 1/1 | yes | 3104 | - |
| unreadable-arguments-then-valid | recovery | 3 | 2 (1) | 0 | - | 26161 | 2/2 | yes | 1159 | - |
| tool-call-written-as-prose | recovery | 3 | 1 (1) | 0 | - | 26067 | 1/1 | yes | 669 | - |
| unknown-tool-is-refused | recovery | 3 | 2 (1) | 0 | - | 26441 | 1/1 | yes | 654 | - |
| integrate-antiderivative | calculus | 2 | 1 (1) | 0 | - | 17348 | 1/1 | yes | 658 | - |
| matrix-expansion-noncommuting | matrix | 2 | 1 (1) | 0 | - | 22170 | 0/2 | yes | 476 | 466 |
| ode-first-order | ode | 2 | 1 (1) | 0 | - | 27284 | 0/1 | yes | 543 | 563 |
| vector-divergence-product | vector | 2 | 1 (1) | 0 | - | 31798 | 0/1 | yes | 675 | 465 |
| vector-distributional-divergence | vector | 2 | 1 (1) | 0 | - | 35592 | 1/2 | yes | 817 | 466 |
