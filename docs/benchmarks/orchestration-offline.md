# Offline orchestration suite

The real agent loop, tools and claim check with a replayed model (no network). Tokens are estimates (`budget.estimate_tokens`, deliberately pessimistic); timings are one run on one machine.

26/26 trajectories end mathematically correct · 60 model calls · 36 tool calls (2 invalid, 1 unknown names) · ≈721957 tokens in, ≈2202 out · 7 unbacked claims

| Trajectory | Category | Model calls | Tool calls (ok) | Routed | Without a model call | Tokens in | Evidence | Math ok | Tool ms | Claim ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wedge-original-failure | notation | 4 | 3 (1) | 0 | 2 refused | 49393 | 4/4 | yes | 2666 | 410 |
| wedge-xdot-first-call | notation | 2 | 1 (1) | 0 | - | 29170 | 4/4 | yes | 882 | 412 |
| wedge-wrong-format-recovered | recovery | 2 | 1 (1) | 0 | 1 re-read | 29198 | 4/4 | yes | 1304 | 421 |
| wedge-shape-errors-repaired | recovery | 2 | 1 (1) | 0 | 1 adjusted | 29240 | 4/4 | yes | 891 | 447 |
| ambiguous-spelling-is-refused-then-fixed | notation | 3 | 2 (1) | 0 | - | 43088 | - | yes | 925 | - |
| equivalent-failed-call-is-not-rerun | recovery | 3 | 2 (0) | 0 | 1 refused | 25734 | - | yes | 811 | - |
| unsupported-is-not-retried | recovery | 3 | 2 (0) | 0 | 1 refused | 25772 | - | yes | 407 | - |
| wedge-routed-xdot | notation | 1 | 1 (1) | 1 | - | 15052 | 4/4 | yes | 879 | 415 |
| pendulum-original-failure | notation | 4 | 3 (2) | 0 | 1 refused | 59866 | 4/4 | yes | 1782 | 948 |
| pendulum-routed-latex | lagrangian | 1 | 1 (1) | 1 | - | 14923 | 4/4 | yes | 891 | 966 |
| pendulum-routed-text-velocities | notation | 1 | 1 (1) | 1 | - | 14913 | 4/4 | yes | 904 | 976 |
| pendulum-model-calls-with-text | notation | 2 | 1 (1) | 0 | - | 28990 | 4/4 | yes | 888 | 950 |
| pendulum-assignment-and-greek | notation | 2 | 1 (1) | 0 | 1 adjusted | 29017 | 4/4 | yes | 889 | 955 |
| pendulum-wrong-sign-stays-unbacked | lagrangian | 2 | 1 (1) | 0 | - | 29004 | 3/4 | yes | 905 | 1505 |
| cyclic-oscillator | conserved | 2 | 1 (1) | 0 | - | 28672 | 1/1 | yes | 472 | 471 |
| simplify-with-functions-prefix | algebra | 2 | 1 (1) | 0 | - | 17214 | 1/2 | yes | 1035 | 411 |
| simplify-text-syntax-then-fix | notation | 3 | 2 (2) | 0 | 1 re-read | 26250 | 1/1 | yes | 4559 | - |
| simplify-text-syntax-recovered | recovery | 2 | 1 (1) | 0 | 1 re-read | 17229 | 1/1 | yes | 2701 | - |
| unreadable-arguments-then-valid | recovery | 3 | 2 (1) | 0 | - | 25791 | 1/2 | yes | 1021 | 415 |
| tool-call-written-as-prose | recovery | 3 | 1 (1) | 0 | - | 25698 | 1/1 | yes | 609 | - |
| unknown-tool-is-refused | recovery | 3 | 2 (1) | 0 | - | 26042 | 1/1 | yes | 593 | - |
| integrate-antiderivative | calculus | 2 | 1 (1) | 0 | - | 17101 | 1/1 | yes | 593 | - |
| matrix-expansion-noncommuting | matrix | 2 | 1 (1) | 0 | - | 21923 | 0/2 | yes | 431 | 430 |
| ode-first-order | ode | 2 | 1 (1) | 0 | - | 27037 | 0/1 | yes | 482 | 563 |
| vector-divergence-product | vector | 2 | 1 (1) | 0 | - | 31324 | 0/1 | yes | 589 | 406 |
| vector-distributional-divergence | vector | 2 | 1 (1) | 0 | - | 34316 | 1/2 | yes | 723 | 427 |
