# Offline orchestration suite

The real agent loop, tools and claim check with a replayed model (no network). Tokens are estimates (`budget.estimate_tokens`, deliberately pessimistic); timings are one run on one machine.

26/26 trajectories end mathematically correct · 60 model calls · 36 tool calls (2 invalid, 1 unknown names) · ≈741833 tokens in, ≈2202 out · 5 unbacked claims

| Trajectory | Category | Model calls | Tool calls (ok) | Routed | Without a model call | Tokens in | Evidence | Math ok | Tool ms | Claim ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wedge-original-failure | notation | 4 | 3 (1) | 0 | 2 refused | 51311 | 4/4 | yes | 1224 | 238 |
| wedge-xdot-first-call | notation | 2 | 1 (1) | 0 | - | 30270 | 4/4 | yes | 683 | 239 |
| wedge-wrong-format-recovered | recovery | 2 | 1 (1) | 0 | 1 re-read | 30297 | 4/4 | yes | 941 | 251 |
| wedge-shape-errors-repaired | recovery | 2 | 1 (1) | 0 | 1 adjusted | 30340 | 4/4 | yes | 701 | 241 |
| ambiguous-spelling-is-refused-then-fixed | notation | 3 | 2 (1) | 0 | - | 44716 | - | yes | 549 | - |
| equivalent-failed-call-is-not-rerun | recovery | 3 | 2 (0) | 0 | 1 refused | 26054 | - | yes | 483 | - |
| unsupported-is-not-retried | recovery | 3 | 2 (0) | 0 | 1 refused | 26090 | - | yes | 235 | - |
| wedge-routed-xdot | notation | 1 | 1 (1) | 1 | - | 15624 | 4/4 | yes | 694 | 236 |
| pendulum-original-failure | notation | 4 | 3 (2) | 0 | 1 refused | 62152 | 4/4 | yes | 1434 | 778 |
| pendulum-routed-latex | lagrangian | 1 | 1 (1) | 1 | - | 15495 | 4/4 | yes | 702 | 770 |
| pendulum-routed-text-velocities | notation | 1 | 1 (1) | 1 | - | 15485 | 4/4 | yes | 722 | 763 |
| pendulum-model-calls-with-text | notation | 2 | 1 (1) | 0 | - | 30090 | 4/4 | yes | 702 | 777 |
| pendulum-assignment-and-greek | notation | 2 | 1 (1) | 0 | 1 adjusted | 30117 | 4/4 | yes | 689 | 776 |
| pendulum-wrong-sign-stays-unbacked | lagrangian | 2 | 1 (1) | 0 | - | 30104 | 3/4 | yes | 723 | 1381 |
| cyclic-oscillator | conserved | 2 | 1 (1) | 0 | - | 29772 | 1/1 | yes | 300 | 304 |
| simplify-with-functions-prefix | algebra | 2 | 1 (1) | 0 | - | 17423 | 2/2 | yes | 674 | - |
| simplify-text-syntax-then-fix | notation | 3 | 2 (2) | 0 | 1 re-read | 26565 | 1/1 | yes | 3499 | - |
| simplify-text-syntax-recovered | recovery | 2 | 1 (1) | 0 | 1 re-read | 17439 | 1/1 | yes | 2045 | - |
| unreadable-arguments-then-valid | recovery | 3 | 2 (1) | 0 | - | 26105 | 2/2 | yes | 639 | - |
| tool-call-written-as-prose | recovery | 3 | 1 (1) | 0 | - | 26010 | 1/1 | yes | 402 | - |
| unknown-tool-is-refused | recovery | 3 | 2 (1) | 0 | - | 26372 | 1/1 | yes | 393 | - |
| integrate-antiderivative | calculus | 2 | 1 (1) | 0 | - | 17310 | 1/1 | yes | 441 | - |
| matrix-expansion-noncommuting | matrix | 2 | 1 (1) | 0 | - | 22132 | 0/2 | yes | 240 | 227 |
| ode-first-order | ode | 2 | 1 (1) | 0 | - | 27246 | 0/1 | yes | 288 | 310 |
| vector-divergence-product | vector | 2 | 1 (1) | 0 | - | 31760 | 0/1 | yes | 401 | 226 |
| vector-distributional-divergence | vector | 2 | 1 (1) | 0 | - | 35554 | 1/2 | yes | 536 | 243 |
