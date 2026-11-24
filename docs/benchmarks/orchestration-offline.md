# Offline orchestration suite

The real agent loop, tools and claim check with a replayed model (no network). Tokens are estimates (`budget.estimate_tokens`, deliberately pessimistic); timings are one run on one machine.

26/26 trajectories end mathematically correct · 60 model calls · 36 tool calls (2 invalid, 1 unknown names) · ≈737165 tokens in, ≈2202 out · 5 unbacked claims

| Trajectory | Category | Model calls | Tool calls (ok) | Routed | Without a model call | Tokens in | Evidence | Math ok | Tool ms | Claim ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wedge-original-failure | notation | 4 | 3 (1) | 0 | 2 refused | 50937 | 4/4 | yes | 2882 | 499 |
| wedge-xdot-first-call | notation | 2 | 1 (1) | 0 | - | 30090 | 4/4 | yes | 985 | 499 |
| wedge-wrong-format-recovered | recovery | 2 | 1 (1) | 0 | 1 re-read | 30118 | 4/4 | yes | 1488 | 465 |
| wedge-shape-errors-repaired | recovery | 2 | 1 (1) | 0 | 1 adjusted | 30160 | 4/4 | yes | 952 | 459 |
| ambiguous-spelling-is-refused-then-fixed | notation | 3 | 2 (1) | 0 | - | 44469 | - | yes | 1009 | - |
| equivalent-failed-call-is-not-rerun | recovery | 3 | 2 (0) | 0 | 1 refused | 25877 | - | yes | 910 | - |
| unsupported-is-not-retried | recovery | 3 | 2 (0) | 0 | 1 refused | 25913 | - | yes | 450 | - |
| wedge-routed-xdot | notation | 1 | 1 (1) | 1 | - | 15512 | 4/4 | yes | 943 | 452 |
| pendulum-original-failure | notation | 4 | 3 (2) | 0 | 1 refused | 61706 | 4/4 | yes | 1917 | 1064 |
| pendulum-routed-latex | lagrangian | 1 | 1 (1) | 1 | - | 15383 | 4/4 | yes | 1017 | 1068 |
| pendulum-routed-text-velocities | notation | 1 | 1 (1) | 1 | - | 15373 | 4/4 | yes | 1004 | 1060 |
| pendulum-model-calls-with-text | notation | 2 | 1 (1) | 0 | - | 29910 | 4/4 | yes | 1022 | 1061 |
| pendulum-assignment-and-greek | notation | 2 | 1 (1) | 0 | 1 adjusted | 29938 | 4/4 | yes | 1046 | 1038 |
| pendulum-wrong-sign-stays-unbacked | lagrangian | 2 | 1 (1) | 0 | - | 29924 | 3/4 | yes | 1023 | 1678 |
| cyclic-oscillator | conserved | 2 | 1 (1) | 0 | - | 29593 | 1/1 | yes | 530 | 558 |
| simplify-with-functions-prefix | algebra | 2 | 1 (1) | 0 | - | 17305 | 2/2 | yes | 1123 | - |
| simplify-text-syntax-then-fix | notation | 3 | 2 (2) | 0 | 1 re-read | 26388 | 1/1 | yes | 5152 | - |
| simplify-text-syntax-recovered | recovery | 2 | 1 (1) | 0 | 1 re-read | 17321 | 1/1 | yes | 2978 | - |
| unreadable-arguments-then-valid | recovery | 3 | 2 (1) | 0 | - | 25928 | 2/2 | yes | 1127 | - |
| tool-call-written-as-prose | recovery | 3 | 1 (1) | 0 | - | 25833 | 1/1 | yes | 686 | - |
| unknown-tool-is-refused | recovery | 3 | 2 (1) | 0 | - | 26187 | 1/1 | yes | 717 | - |
| integrate-antiderivative | calculus | 2 | 1 (1) | 0 | - | 17192 | 1/1 | yes | 676 | - |
| matrix-expansion-noncommuting | matrix | 2 | 1 (1) | 0 | - | 22014 | 0/2 | yes | 497 | 483 |
| ode-first-order | ode | 2 | 1 (1) | 0 | - | 27128 | 0/1 | yes | 541 | 553 |
| vector-divergence-product | vector | 2 | 1 (1) | 0 | - | 31642 | 0/1 | yes | 650 | 464 |
| vector-distributional-divergence | vector | 2 | 1 (1) | 0 | - | 35324 | 1/2 | yes | 784 | 460 |
