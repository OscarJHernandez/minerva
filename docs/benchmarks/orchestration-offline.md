# Offline orchestration suite

The real agent loop, tools and claim check with a replayed model (no network). Tokens are estimates (`budget.estimate_tokens`, deliberately pessimistic); timings are one run on one machine.

26/26 trajectories end mathematically correct · 60 model calls · 36 tool calls (2 invalid, 1 unknown names) · ≈738314 tokens in, ≈2202 out · 5 unbacked claims

| Trajectory | Category | Model calls | Tool calls (ok) | Routed | Without a model call | Tokens in | Evidence | Math ok | Tool ms | Claim ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wedge-original-failure | notation | 4 | 3 (1) | 0 | 2 refused | 51104 | 4/4 | yes | 3248 | 512 |
| wedge-xdot-first-call | notation | 2 | 1 (1) | 0 | - | 30152 | 4/4 | yes | 1038 | 541 |
| wedge-wrong-format-recovered | recovery | 2 | 1 (1) | 0 | 1 re-read | 30179 | 4/4 | yes | 1563 | 547 |
| wedge-shape-errors-repaired | recovery | 2 | 1 (1) | 0 | 1 adjusted | 30222 | 4/4 | yes | 1068 | 535 |
| ambiguous-spelling-is-refused-then-fixed | notation | 3 | 2 (1) | 0 | - | 44539 | - | yes | 1157 | - |
| equivalent-failed-call-is-not-rerun | recovery | 3 | 2 (0) | 0 | 1 refused | 25877 | - | yes | 1025 | - |
| unsupported-is-not-retried | recovery | 3 | 2 (0) | 0 | 1 refused | 25913 | - | yes | 503 | - |
| wedge-routed-xdot | notation | 1 | 1 (1) | 1 | - | 15565 | 4/4 | yes | 1053 | 556 |
| pendulum-original-failure | notation | 4 | 3 (2) | 0 | 1 refused | 61916 | 4/4 | yes | 2182 | 1166 |
| pendulum-routed-latex | lagrangian | 1 | 1 (1) | 1 | - | 15436 | 4/4 | yes | 1097 | 1181 |
| pendulum-routed-text-velocities | notation | 1 | 1 (1) | 1 | - | 15426 | 4/4 | yes | 1056 | 1136 |
| pendulum-model-calls-with-text | notation | 2 | 1 (1) | 0 | - | 29972 | 4/4 | yes | 1005 | 1083 |
| pendulum-assignment-and-greek | notation | 2 | 1 (1) | 0 | 1 adjusted | 29999 | 4/4 | yes | 1073 | 1177 |
| pendulum-wrong-sign-stays-unbacked | lagrangian | 2 | 1 (1) | 0 | - | 29986 | 3/4 | yes | 1107 | 1817 |
| cyclic-oscillator | conserved | 2 | 1 (1) | 0 | - | 29654 | 1/1 | yes | 637 | 583 |
| simplify-with-functions-prefix | algebra | 2 | 1 (1) | 0 | - | 17305 | 2/2 | yes | 1253 | - |
| simplify-text-syntax-then-fix | notation | 3 | 2 (2) | 0 | 1 re-read | 26388 | 1/1 | yes | 5555 | - |
| simplify-text-syntax-recovered | recovery | 2 | 1 (1) | 0 | 1 re-read | 17321 | 1/1 | yes | 3206 | - |
| unreadable-arguments-then-valid | recovery | 3 | 2 (1) | 0 | - | 25928 | 2/2 | yes | 1251 | - |
| tool-call-written-as-prose | recovery | 3 | 1 (1) | 0 | - | 25833 | 1/1 | yes | 782 | - |
| unknown-tool-is-refused | recovery | 3 | 2 (1) | 0 | - | 26187 | 1/1 | yes | 720 | - |
| integrate-antiderivative | calculus | 2 | 1 (1) | 0 | - | 17192 | 1/1 | yes | 719 | - |
| matrix-expansion-noncommuting | matrix | 2 | 1 (1) | 0 | - | 22014 | 0/2 | yes | 514 | 473 |
| ode-first-order | ode | 2 | 1 (1) | 0 | - | 27128 | 0/1 | yes | 580 | 576 |
| vector-divergence-product | vector | 2 | 1 (1) | 0 | - | 31642 | 0/1 | yes | 802 | 521 |
| vector-distributional-divergence | vector | 2 | 1 (1) | 0 | - | 35436 | 1/2 | yes | 928 | 537 |
