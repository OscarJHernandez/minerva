# Offline orchestration suite

The real agent loop, tools and claim check with a replayed model (no network). Tokens are estimates (`budget.estimate_tokens`, deliberately pessimistic); timings are one run on one machine.

26/26 trajectories end mathematically correct · 60 model calls · 36 tool calls (2 invalid, 1 unknown names) · ≈742979 tokens in, ≈2202 out · 5 unbacked claims

| Trajectory | Category | Model calls | Tool calls (ok) | Routed | Without a model call | Tokens in | Evidence | Math ok | Tool ms | Claim ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wedge-original-failure | notation | 4 | 3 (1) | 0 | 2 refused | 51376 | 4/4 | yes | 2805 | 505 |
| wedge-xdot-first-call | notation | 2 | 1 (1) | 0 | - | 30308 | 4/4 | yes | 825 | 496 |
| wedge-wrong-format-recovered | recovery | 2 | 1 (1) | 0 | 1 re-read | 30335 | 4/4 | yes | 1396 | 542 |
| wedge-shape-errors-repaired | recovery | 2 | 1 (1) | 0 | 1 adjusted | 30378 | 4/4 | yes | 888 | 466 |
| ambiguous-spelling-is-refused-then-fixed | notation | 3 | 2 (1) | 0 | - | 44771 | - | yes | 1070 | - |
| equivalent-failed-call-is-not-rerun | recovery | 3 | 2 (0) | 0 | 1 refused | 26110 | - | yes | 1007 | - |
| unsupported-is-not-retried | recovery | 3 | 2 (0) | 0 | 1 refused | 26147 | - | yes | 495 | - |
| wedge-routed-xdot | notation | 1 | 1 (1) | 1 | - | 15643 | 4/4 | yes | 838 | 494 |
| pendulum-original-failure | notation | 4 | 3 (2) | 0 | 1 refused | 62226 | 4/4 | yes | 1795 | 1011 |
| pendulum-routed-latex | lagrangian | 1 | 1 (1) | 1 | - | 15514 | 4/4 | yes | 915 | 1005 |
| pendulum-routed-text-velocities | notation | 1 | 1 (1) | 1 | - | 15504 | 4/4 | yes | 924 | 1035 |
| pendulum-model-calls-with-text | notation | 2 | 1 (1) | 0 | - | 30128 | 4/4 | yes | 945 | 1025 |
| pendulum-assignment-and-greek | notation | 2 | 1 (1) | 0 | 1 adjusted | 30155 | 4/4 | yes | 966 | 1109 |
| pendulum-wrong-sign-stays-unbacked | lagrangian | 2 | 1 (1) | 0 | - | 30141 | 3/4 | yes | 930 | 1600 |
| cyclic-oscillator | conserved | 2 | 1 (1) | 0 | - | 29810 | 1/1 | yes | 526 | 537 |
| simplify-with-functions-prefix | algebra | 2 | 1 (1) | 0 | - | 17461 | 2/2 | yes | 1206 | - |
| simplify-text-syntax-then-fix | notation | 3 | 2 (2) | 0 | 1 re-read | 26622 | 1/1 | yes | 5110 | - |
| simplify-text-syntax-recovered | recovery | 2 | 1 (1) | 0 | 1 re-read | 17477 | 1/1 | yes | 3044 | - |
| unreadable-arguments-then-valid | recovery | 3 | 2 (1) | 0 | - | 26161 | 2/2 | yes | 1241 | - |
| tool-call-written-as-prose | recovery | 3 | 1 (1) | 0 | - | 26067 | 1/1 | yes | 705 | - |
| unknown-tool-is-refused | recovery | 3 | 2 (1) | 0 | - | 26453 | 1/1 | yes | 696 | - |
| integrate-antiderivative | calculus | 2 | 1 (1) | 0 | - | 17348 | 1/1 | yes | 686 | - |
| matrix-expansion-noncommuting | matrix | 2 | 1 (1) | 0 | - | 22170 | 0/2 | yes | 531 | 507 |
| ode-first-order | ode | 2 | 1 (1) | 0 | - | 27284 | 0/1 | yes | 544 | 647 |
| vector-divergence-product | vector | 2 | 1 (1) | 0 | - | 31798 | 0/1 | yes | 697 | 487 |
| vector-distributional-divergence | vector | 2 | 1 (1) | 0 | - | 35592 | 1/2 | yes | 829 | 462 |
