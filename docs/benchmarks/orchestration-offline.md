# Offline orchestration suite

The real agent loop, tools and claim check with a replayed model (no network). Tokens are estimates (`budget.estimate_tokens`, deliberately pessimistic); timings are one run on one machine.

26/26 trajectories end mathematically correct · 60 model calls · 36 tool calls (2 invalid, 1 unknown names) · ≈723796 tokens in, ≈2202 out · 5 unbacked claims

| Trajectory | Category | Model calls | Tool calls (ok) | Routed | Without a model call | Tokens in | Evidence | Math ok | Tool ms | Claim ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wedge-original-failure | notation | 4 | 3 (1) | 0 | 2 refused | 49463 | 4/4 | yes | 1240 | 234 |
| wedge-xdot-first-call | notation | 2 | 1 (1) | 0 | - | 29217 | 4/4 | yes | 725 | 244 |
| wedge-wrong-format-recovered | recovery | 2 | 1 (1) | 0 | 1 re-read | 29244 | 4/4 | yes | 961 | 237 |
| wedge-shape-errors-repaired | recovery | 2 | 1 (1) | 0 | 1 adjusted | 29287 | 4/4 | yes | 722 | 246 |
| ambiguous-spelling-is-refused-then-fixed | notation | 3 | 2 (1) | 0 | - | 43158 | - | yes | 591 | - |
| equivalent-failed-call-is-not-rerun | recovery | 3 | 2 (0) | 0 | 1 refused | 25805 | - | yes | 468 | - |
| unsupported-is-not-retried | recovery | 3 | 2 (0) | 0 | 1 refused | 25842 | - | yes | 229 | - |
| wedge-routed-xdot | notation | 1 | 1 (1) | 1 | - | 15075 | 4/4 | yes | 713 | 246 |
| pendulum-original-failure | notation | 4 | 3 (2) | 0 | 1 refused | 59959 | 4/4 | yes | 1431 | 789 |
| pendulum-routed-latex | lagrangian | 1 | 1 (1) | 1 | - | 14947 | 4/4 | yes | 770 | 784 |
| pendulum-routed-text-velocities | notation | 1 | 1 (1) | 1 | - | 14937 | 4/4 | yes | 765 | 807 |
| pendulum-model-calls-with-text | notation | 2 | 1 (1) | 0 | - | 29038 | 4/4 | yes | 729 | 825 |
| pendulum-assignment-and-greek | notation | 2 | 1 (1) | 0 | 1 adjusted | 29065 | 4/4 | yes | 738 | 819 |
| pendulum-wrong-sign-stays-unbacked | lagrangian | 2 | 1 (1) | 0 | - | 29051 | 3/4 | yes | 751 | 1424 |
| cyclic-oscillator | conserved | 2 | 1 (1) | 0 | - | 28719 | 1/1 | yes | 300 | 297 |
| simplify-with-functions-prefix | algebra | 2 | 1 (1) | 0 | - | 17261 | 2/2 | yes | 667 | - |
| simplify-text-syntax-then-fix | notation | 3 | 2 (2) | 0 | 1 re-read | 26322 | 1/1 | yes | 3740 | - |
| simplify-text-syntax-recovered | recovery | 2 | 1 (1) | 0 | 1 re-read | 17277 | 1/1 | yes | 2033 | - |
| unreadable-arguments-then-valid | recovery | 3 | 2 (1) | 0 | - | 25861 | 2/2 | yes | 656 | - |
| tool-call-written-as-prose | recovery | 3 | 1 (1) | 0 | - | 25767 | 1/1 | yes | 407 | - |
| unknown-tool-is-refused | recovery | 3 | 2 (1) | 0 | - | 26112 | 1/1 | yes | 383 | - |
| integrate-antiderivative | calculus | 2 | 1 (1) | 0 | - | 17148 | 1/1 | yes | 398 | - |
| matrix-expansion-noncommuting | matrix | 2 | 1 (1) | 0 | - | 21970 | 0/2 | yes | 248 | 232 |
| ode-first-order | ode | 2 | 1 (1) | 0 | - | 27084 | 0/1 | yes | 296 | 389 |
| vector-divergence-product | vector | 2 | 1 (1) | 0 | - | 31598 | 0/1 | yes | 419 | 231 |
| vector-distributional-divergence | vector | 2 | 1 (1) | 0 | - | 34589 | 1/2 | yes | 585 | 234 |
