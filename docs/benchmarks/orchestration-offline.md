# Offline orchestration suite

The real agent loop, tools and claim check with a replayed model (no network). Tokens are estimates (`budget.estimate_tokens`, deliberately pessimistic); timings are one run on one machine.

26/26 trajectories end mathematically correct · 60 model calls · 36 tool calls (2 invalid, 1 unknown names) · ≈737277 tokens in, ≈2202 out · 5 unbacked claims

| Trajectory | Category | Model calls | Tool calls (ok) | Routed | Without a model call | Tokens in | Evidence | Math ok | Tool ms | Claim ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wedge-original-failure | notation | 4 | 3 (1) | 0 | 2 refused | 50937 | 4/4 | yes | 2672 | 450 |
| wedge-xdot-first-call | notation | 2 | 1 (1) | 0 | - | 30090 | 4/4 | yes | 916 | 427 |
| wedge-wrong-format-recovered | recovery | 2 | 1 (1) | 0 | 1 re-read | 30118 | 4/4 | yes | 1349 | 474 |
| wedge-shape-errors-repaired | recovery | 2 | 1 (1) | 0 | 1 adjusted | 30160 | 4/4 | yes | 935 | 417 |
| ambiguous-spelling-is-refused-then-fixed | notation | 3 | 2 (1) | 0 | - | 44469 | - | yes | 894 | - |
| equivalent-failed-call-is-not-rerun | recovery | 3 | 2 (0) | 0 | 1 refused | 25877 | - | yes | 807 | - |
| unsupported-is-not-retried | recovery | 3 | 2 (0) | 0 | 1 refused | 25913 | - | yes | 409 | - |
| wedge-routed-xdot | notation | 1 | 1 (1) | 1 | - | 15512 | 4/4 | yes | 856 | 409 |
| pendulum-original-failure | notation | 4 | 3 (2) | 0 | 1 refused | 61706 | 4/4 | yes | 1768 | 965 |
| pendulum-routed-latex | lagrangian | 1 | 1 (1) | 1 | - | 15383 | 4/4 | yes | 886 | 924 |
| pendulum-routed-text-velocities | notation | 1 | 1 (1) | 1 | - | 15373 | 4/4 | yes | 865 | 937 |
| pendulum-model-calls-with-text | notation | 2 | 1 (1) | 0 | - | 29910 | 4/4 | yes | 866 | 937 |
| pendulum-assignment-and-greek | notation | 2 | 1 (1) | 0 | 1 adjusted | 29938 | 4/4 | yes | 907 | 981 |
| pendulum-wrong-sign-stays-unbacked | lagrangian | 2 | 1 (1) | 0 | - | 29924 | 3/4 | yes | 918 | 1592 |
| cyclic-oscillator | conserved | 2 | 1 (1) | 0 | - | 29593 | 1/1 | yes | 533 | 530 |
| simplify-with-functions-prefix | algebra | 2 | 1 (1) | 0 | - | 17305 | 2/2 | yes | 1067 | - |
| simplify-text-syntax-then-fix | notation | 3 | 2 (2) | 0 | 1 re-read | 26388 | 1/1 | yes | 4774 | - |
| simplify-text-syntax-recovered | recovery | 2 | 1 (1) | 0 | 1 re-read | 17321 | 1/1 | yes | 2719 | - |
| unreadable-arguments-then-valid | recovery | 3 | 2 (1) | 0 | - | 25928 | 2/2 | yes | 1008 | - |
| tool-call-written-as-prose | recovery | 3 | 1 (1) | 0 | - | 25833 | 1/1 | yes | 589 | - |
| unknown-tool-is-refused | recovery | 3 | 2 (1) | 0 | - | 26187 | 1/1 | yes | 575 | - |
| integrate-antiderivative | calculus | 2 | 1 (1) | 0 | - | 17192 | 1/1 | yes | 573 | - |
| matrix-expansion-noncommuting | matrix | 2 | 1 (1) | 0 | - | 22014 | 0/2 | yes | 413 | 405 |
| ode-first-order | ode | 2 | 1 (1) | 0 | - | 27128 | 0/1 | yes | 476 | 490 |
| vector-divergence-product | vector | 2 | 1 (1) | 0 | - | 31642 | 0/1 | yes | 579 | 400 |
| vector-distributional-divergence | vector | 2 | 1 (1) | 0 | - | 35436 | 1/2 | yes | 713 | 399 |
