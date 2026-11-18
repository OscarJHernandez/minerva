# Offline orchestration suite

The real agent loop, tools and claim check with a replayed model (no network). Tokens are estimates (`budget.estimate_tokens`, deliberately pessimistic); timings are one run on one machine.

26/26 trajectories end mathematically correct · 60 model calls · 36 tool calls (2 invalid, 1 unknown names) · ≈725125 tokens in, ≈2202 out · 5 unbacked claims

| Trajectory | Category | Model calls | Tool calls (ok) | Routed | Without a model call | Tokens in | Evidence | Math ok | Tool ms | Claim ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wedge-original-failure | notation | 4 | 3 (1) | 0 | 2 refused | 49542 | 4/4 | yes | 2530 | 385 |
| wedge-xdot-first-call | notation | 2 | 1 (1) | 0 | - | 29262 | 4/4 | yes | 840 | 385 |
| wedge-wrong-format-recovered | recovery | 2 | 1 (1) | 0 | 1 re-read | 29289 | 4/4 | yes | 1209 | 384 |
| wedge-shape-errors-repaired | recovery | 2 | 1 (1) | 0 | 1 adjusted | 29332 | 4/4 | yes | 829 | 390 |
| ambiguous-spelling-is-refused-then-fixed | notation | 3 | 2 (1) | 0 | - | 43225 | - | yes | 847 | - |
| equivalent-failed-call-is-not-rerun | recovery | 3 | 2 (0) | 0 | 1 refused | 25872 | - | yes | 764 | - |
| unsupported-is-not-retried | recovery | 3 | 2 (0) | 0 | 1 refused | 25908 | - | yes | 383 | - |
| wedge-routed-xdot | notation | 1 | 1 (1) | 1 | - | 15098 | 4/4 | yes | 831 | 384 |
| pendulum-original-failure | notation | 4 | 3 (2) | 0 | 1 refused | 60047 | 4/4 | yes | 1721 | 900 |
| pendulum-routed-latex | lagrangian | 1 | 1 (1) | 1 | - | 14969 | 4/4 | yes | 858 | 892 |
| pendulum-routed-text-velocities | notation | 1 | 1 (1) | 1 | - | 14959 | 4/4 | yes | 842 | 898 |
| pendulum-model-calls-with-text | notation | 2 | 1 (1) | 0 | - | 29082 | 4/4 | yes | 848 | 897 |
| pendulum-assignment-and-greek | notation | 2 | 1 (1) | 0 | 1 adjusted | 29109 | 4/4 | yes | 845 | 893 |
| pendulum-wrong-sign-stays-unbacked | lagrangian | 2 | 1 (1) | 0 | - | 29095 | 3/4 | yes | 849 | 1449 |
| cyclic-oscillator | conserved | 2 | 1 (1) | 0 | - | 28764 | 1/1 | yes | 446 | 442 |
| simplify-with-functions-prefix | algebra | 2 | 1 (1) | 0 | - | 17305 | 2/2 | yes | 965 | - |
| simplify-text-syntax-then-fix | notation | 3 | 2 (2) | 0 | 1 re-read | 26388 | 1/1 | yes | 4288 | - |
| simplify-text-syntax-recovered | recovery | 2 | 1 (1) | 0 | 1 re-read | 17321 | 1/1 | yes | 2625 | - |
| unreadable-arguments-then-valid | recovery | 3 | 2 (1) | 0 | - | 25928 | 2/2 | yes | 965 | - |
| tool-call-written-as-prose | recovery | 3 | 1 (1) | 0 | - | 25833 | 1/1 | yes | 564 | - |
| unknown-tool-is-refused | recovery | 3 | 2 (1) | 0 | - | 26187 | 1/1 | yes | 557 | - |
| integrate-antiderivative | calculus | 2 | 1 (1) | 0 | - | 17192 | 1/1 | yes | 564 | - |
| matrix-expansion-noncommuting | matrix | 2 | 1 (1) | 0 | - | 22014 | 0/2 | yes | 393 | 380 |
| ode-first-order | ode | 2 | 1 (1) | 0 | - | 27128 | 0/1 | yes | 445 | 530 |
| vector-divergence-product | vector | 2 | 1 (1) | 0 | - | 31642 | 0/1 | yes | 559 | 382 |
| vector-distributional-divergence | vector | 2 | 1 (1) | 0 | - | 34634 | 1/2 | yes | 688 | 385 |
