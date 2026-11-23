# Offline orchestration suite

The real agent loop, tools and claim check with a replayed model (no network). Tokens are estimates (`budget.estimate_tokens`, deliberately pessimistic); timings are one run on one machine.

26/26 trajectories end mathematically correct · 60 model calls · 36 tool calls (2 invalid, 1 unknown names) · ≈736475 tokens in, ≈2202 out · 5 unbacked claims

| Trajectory | Category | Model calls | Tool calls (ok) | Routed | Without a model call | Tokens in | Evidence | Math ok | Tool ms | Claim ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wedge-original-failure | notation | 4 | 3 (1) | 0 | 2 refused | 50937 | 4/4 | yes | 2774 | 437 |
| wedge-xdot-first-call | notation | 2 | 1 (1) | 0 | - | 30090 | 4/4 | yes | 919 | 416 |
| wedge-wrong-format-recovered | recovery | 2 | 1 (1) | 0 | 1 re-read | 30118 | 4/4 | yes | 1311 | 400 |
| wedge-shape-errors-repaired | recovery | 2 | 1 (1) | 0 | 1 adjusted | 30160 | 4/4 | yes | 848 | 398 |
| ambiguous-spelling-is-refused-then-fixed | notation | 3 | 2 (1) | 0 | - | 44469 | - | yes | 879 | - |
| equivalent-failed-call-is-not-rerun | recovery | 3 | 2 (0) | 0 | 1 refused | 25877 | - | yes | 796 | - |
| unsupported-is-not-retried | recovery | 3 | 2 (0) | 0 | 1 refused | 25913 | - | yes | 395 | - |
| wedge-routed-xdot | notation | 1 | 1 (1) | 1 | - | 15512 | 4/4 | yes | 854 | 402 |
| pendulum-original-failure | notation | 4 | 3 (2) | 0 | 1 refused | 61706 | 4/4 | yes | 1710 | 952 |
| pendulum-routed-latex | lagrangian | 1 | 1 (1) | 1 | - | 15383 | 4/4 | yes | 881 | 923 |
| pendulum-routed-text-velocities | notation | 1 | 1 (1) | 1 | - | 15373 | 4/4 | yes | 936 | 929 |
| pendulum-model-calls-with-text | notation | 2 | 1 (1) | 0 | - | 29910 | 4/4 | yes | 931 | 1066 |
| pendulum-assignment-and-greek | notation | 2 | 1 (1) | 0 | 1 adjusted | 29938 | 4/4 | yes | 980 | 976 |
| pendulum-wrong-sign-stays-unbacked | lagrangian | 2 | 1 (1) | 0 | - | 29924 | 3/4 | yes | 854 | 1475 |
| cyclic-oscillator | conserved | 2 | 1 (1) | 0 | - | 29593 | 1/1 | yes | 456 | 452 |
| simplify-with-functions-prefix | algebra | 2 | 1 (1) | 0 | - | 17305 | 2/2 | yes | 976 | - |
| simplify-text-syntax-then-fix | notation | 3 | 2 (2) | 0 | 1 re-read | 26388 | 1/1 | yes | 4400 | - |
| simplify-text-syntax-recovered | recovery | 2 | 1 (1) | 0 | 1 re-read | 17321 | 1/1 | yes | 2586 | - |
| unreadable-arguments-then-valid | recovery | 3 | 2 (1) | 0 | - | 25928 | 2/2 | yes | 995 | - |
| tool-call-written-as-prose | recovery | 3 | 1 (1) | 0 | - | 25833 | 1/1 | yes | 570 | - |
| unknown-tool-is-refused | recovery | 3 | 2 (1) | 0 | - | 26187 | 1/1 | yes | 564 | - |
| integrate-antiderivative | calculus | 2 | 1 (1) | 0 | - | 17192 | 1/1 | yes | 564 | - |
| matrix-expansion-noncommuting | matrix | 2 | 1 (1) | 0 | - | 22014 | 0/2 | yes | 407 | 415 |
| ode-first-order | ode | 2 | 1 (1) | 0 | - | 27128 | 0/1 | yes | 460 | 540 |
| vector-divergence-product | vector | 2 | 1 (1) | 0 | - | 31642 | 0/1 | yes | 569 | 392 |
| vector-distributional-divergence | vector | 2 | 1 (1) | 0 | - | 34634 | 1/2 | yes | 700 | 393 |
