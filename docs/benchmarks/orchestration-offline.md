# Offline orchestration suite

The real agent loop, tools and claim check with a replayed model (no network). Tokens are estimates (`budget.estimate_tokens`, deliberately pessimistic); timings are one run on one machine.

26/26 trajectories end mathematically correct · 60 model calls · 36 tool calls (2 invalid, 1 unknown names) · ≈726812 tokens in, ≈2202 out · 5 unbacked claims

| Trajectory | Category | Model calls | Tool calls (ok) | Routed | Without a model call | Tokens in | Evidence | Math ok | Tool ms | Claim ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wedge-original-failure | notation | 4 | 3 (1) | 0 | 2 refused | 49783 | 4/4 | yes | 2606 | 408 |
| wedge-xdot-first-call | notation | 2 | 1 (1) | 0 | - | 29382 | 4/4 | yes | 867 | 419 |
| wedge-wrong-format-recovered | recovery | 2 | 1 (1) | 0 | 1 re-read | 29410 | 4/4 | yes | 1277 | 409 |
| wedge-shape-errors-repaired | recovery | 2 | 1 (1) | 0 | 1 adjusted | 29452 | 4/4 | yes | 875 | 416 |
| ambiguous-spelling-is-refused-then-fixed | notation | 3 | 2 (1) | 0 | - | 43406 | - | yes | 958 | - |
| equivalent-failed-call-is-not-rerun | recovery | 3 | 2 (0) | 0 | 1 refused | 25872 | - | yes | 818 | - |
| unsupported-is-not-retried | recovery | 3 | 2 (0) | 0 | 1 refused | 25908 | - | yes | 408 | - |
| wedge-routed-xdot | notation | 1 | 1 (1) | 1 | - | 15158 | 4/4 | yes | 871 | 419 |
| pendulum-original-failure | notation | 4 | 3 (2) | 0 | 1 refused | 60290 | 4/4 | yes | 1788 | 948 |
| pendulum-routed-latex | lagrangian | 1 | 1 (1) | 1 | - | 15029 | 4/4 | yes | 891 | 960 |
| pendulum-routed-text-velocities | notation | 1 | 1 (1) | 1 | - | 15019 | 4/4 | yes | 890 | 951 |
| pendulum-model-calls-with-text | notation | 2 | 1 (1) | 0 | - | 29202 | 4/4 | yes | 911 | 987 |
| pendulum-assignment-and-greek | notation | 2 | 1 (1) | 0 | 1 adjusted | 29229 | 4/4 | yes | 934 | 990 |
| pendulum-wrong-sign-stays-unbacked | lagrangian | 2 | 1 (1) | 0 | - | 29216 | 3/4 | yes | 924 | 1547 |
| cyclic-oscillator | conserved | 2 | 1 (1) | 0 | - | 28884 | 1/1 | yes | 491 | 528 |
| simplify-with-functions-prefix | algebra | 2 | 1 (1) | 0 | - | 17305 | 2/2 | yes | 1126 | - |
| simplify-text-syntax-then-fix | notation | 3 | 2 (2) | 0 | 1 re-read | 26388 | 1/1 | yes | 4822 | - |
| simplify-text-syntax-recovered | recovery | 2 | 1 (1) | 0 | 1 re-read | 17321 | 1/1 | yes | 2733 | - |
| unreadable-arguments-then-valid | recovery | 3 | 2 (1) | 0 | - | 25928 | 2/2 | yes | 1039 | - |
| tool-call-written-as-prose | recovery | 3 | 1 (1) | 0 | - | 25833 | 1/1 | yes | 604 | - |
| unknown-tool-is-refused | recovery | 3 | 2 (1) | 0 | - | 26187 | 1/1 | yes | 606 | - |
| integrate-antiderivative | calculus | 2 | 1 (1) | 0 | - | 17192 | 1/1 | yes | 640 | - |
| matrix-expansion-noncommuting | matrix | 2 | 1 (1) | 0 | - | 22014 | 0/2 | yes | 454 | 421 |
| ode-first-order | ode | 2 | 1 (1) | 0 | - | 27128 | 0/1 | yes | 486 | 590 |
| vector-divergence-product | vector | 2 | 1 (1) | 0 | - | 31642 | 0/1 | yes | 601 | 415 |
| vector-distributional-divergence | vector | 2 | 1 (1) | 0 | - | 34634 | 1/2 | yes | 732 | 416 |
