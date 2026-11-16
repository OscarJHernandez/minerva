# Offline orchestration suite

The real agent loop, tools and claim check with a replayed model (no network). Tokens are estimates (`budget.estimate_tokens`, deliberately pessimistic); timings are one run on one machine.

25/25 trajectories end mathematically correct · 58 model calls · 35 tool calls (2 invalid, 1 unknown names) · ≈686478 tokens in, ≈2092 out · 6 unbacked claims

| Trajectory | Category | Model calls | Tool calls (ok) | Routed | Without a model call | Tokens in | Evidence | Math ok | Tool ms | Claim ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wedge-original-failure | notation | 4 | 3 (1) | 0 | 2 refused | 49393 | 4/4 | yes | 2847 | 462 |
| wedge-xdot-first-call | notation | 2 | 1 (1) | 0 | - | 29170 | 4/4 | yes | 946 | 435 |
| wedge-wrong-format-recovered | recovery | 2 | 1 (1) | 0 | 1 re-read | 29198 | 4/4 | yes | 1394 | 443 |
| wedge-shape-errors-repaired | recovery | 2 | 1 (1) | 0 | 1 adjusted | 29240 | 4/4 | yes | 932 | 442 |
| ambiguous-spelling-is-refused-then-fixed | notation | 3 | 2 (1) | 0 | - | 43088 | - | yes | 966 | - |
| equivalent-failed-call-is-not-rerun | recovery | 3 | 2 (0) | 0 | 1 refused | 25734 | - | yes | 850 | - |
| unsupported-is-not-retried | recovery | 3 | 2 (0) | 0 | 1 refused | 25772 | - | yes | 480 | - |
| wedge-routed-xdot | notation | 1 | 1 (1) | 1 | - | 15052 | 4/4 | yes | 0 | 431 |
| pendulum-original-failure | notation | 4 | 3 (2) | 0 | 1 refused | 59866 | 4/4 | yes | 1904 | 1031 |
| pendulum-routed-latex | lagrangian | 1 | 1 (1) | 1 | - | 14923 | 4/4 | yes | 0 | 1011 |
| pendulum-routed-text-velocities | notation | 1 | 1 (1) | 1 | - | 14913 | 4/4 | yes | 0 | 1016 |
| pendulum-model-calls-with-text | notation | 2 | 1 (1) | 0 | - | 28990 | 4/4 | yes | 935 | 1008 |
| pendulum-assignment-and-greek | notation | 2 | 1 (1) | 0 | 1 adjusted | 29017 | 4/4 | yes | 943 | 1067 |
| pendulum-wrong-sign-stays-unbacked | lagrangian | 2 | 1 (1) | 0 | - | 29004 | 3/4 | yes | 957 | 1624 |
| cyclic-oscillator | conserved | 2 | 1 (1) | 0 | - | 28672 | 1/1 | yes | 501 | 500 |
| simplify-with-functions-prefix | algebra | 2 | 1 (1) | 0 | - | 17214 | 1/2 | yes | 1070 | 433 |
| simplify-text-syntax-then-fix | notation | 3 | 2 (2) | 0 | 1 re-read | 26250 | 1/1 | yes | 4837 | - |
| simplify-text-syntax-recovered | recovery | 2 | 1 (1) | 0 | 1 re-read | 17229 | 1/1 | yes | 2858 | - |
| unreadable-arguments-then-valid | recovery | 3 | 2 (1) | 0 | - | 25791 | 1/2 | yes | 1061 | 423 |
| tool-call-written-as-prose | recovery | 3 | 1 (1) | 0 | - | 25698 | 1/1 | yes | 621 | - |
| unknown-tool-is-refused | recovery | 3 | 2 (1) | 0 | - | 26042 | 1/1 | yes | 607 | - |
| integrate-antiderivative | calculus | 2 | 1 (1) | 0 | - | 17101 | 1/1 | yes | 614 | - |
| matrix-expansion-noncommuting | matrix | 2 | 1 (1) | 0 | - | 21923 | 0/2 | yes | 435 | 424 |
| ode-first-order | ode | 2 | 1 (1) | 0 | - | 27037 | 0/1 | yes | 492 | 587 |
| vector-divergence-product | vector | 2 | 1 (1) | 0 | - | 30161 | 0/1 | yes | 611 | 420 |
