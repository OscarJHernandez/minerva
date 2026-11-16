# Offline orchestration suite

The real agent loop, tools and claim check with a replayed model (no network). Tokens are estimates (`budget.estimate_tokens`, deliberately pessimistic); timings are one run on one machine.

19/19 trajectories end mathematically correct · 43 model calls · 26 tool calls (2 invalid, 1 unknown names) · ≈515881 tokens in, ≈1646 out · 6 unbacked claims

| Trajectory | Category | Model calls | Tool calls (ok) | Routed | Tokens in | Evidence | Math ok | Tool ms | Claim ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wedge-original-failure | notation | 4 | 3 (1) | 0 | 49393 | 4/4 | yes | 2763 | 420 |
| wedge-xdot-first-call | notation | 2 | 1 (1) | 0 | 29170 | 4/4 | yes | 889 | 394 |
| wedge-routed-xdot | notation | 1 | 1 (1) | 1 | 15052 | 4/4 | yes | 0 | 422 |
| pendulum-original-failure | notation | 4 | 3 (2) | 0 | 59866 | 4/4 | yes | 1899 | 1008 |
| pendulum-routed-latex | lagrangian | 1 | 1 (1) | 1 | 14923 | 4/4 | yes | 0 | 986 |
| pendulum-routed-text-velocities | notation | 1 | 1 (1) | 1 | 14913 | 4/4 | yes | 0 | 1008 |
| pendulum-model-calls-with-text | notation | 2 | 1 (1) | 0 | 28990 | 4/4 | yes | 900 | 1037 |
| pendulum-assignment-and-greek | notation | 2 | 1 (1) | 0 | 29017 | 4/4 | yes | 958 | 1026 |
| pendulum-wrong-sign-stays-unbacked | lagrangian | 2 | 1 (1) | 0 | 29004 | 3/4 | yes | 937 | 1603 |
| cyclic-oscillator | conserved | 2 | 1 (1) | 0 | 28672 | 1/1 | yes | 507 | 490 |
| simplify-with-functions-prefix | algebra | 2 | 1 (1) | 0 | 17214 | 1/2 | yes | 964 | 428 |
| simplify-text-syntax-then-fix | notation | 3 | 2 (1) | 0 | 25914 | 1/1 | yes | 2409 | - |
| unreadable-arguments-then-valid | recovery | 3 | 2 (1) | 0 | 25791 | 1/2 | yes | 1099 | 434 |
| tool-call-written-as-prose | recovery | 3 | 1 (1) | 0 | 25698 | 1/1 | yes | 661 | - |
| unknown-tool-is-refused | recovery | 3 | 2 (1) | 0 | 26042 | 1/1 | yes | 572 | - |
| integrate-antiderivative | calculus | 2 | 1 (1) | 0 | 17101 | 1/1 | yes | 633 | - |
| matrix-expansion-noncommuting | matrix | 2 | 1 (1) | 0 | 21923 | 0/2 | yes | 422 | 425 |
| ode-first-order | ode | 2 | 1 (1) | 0 | 27037 | 0/1 | yes | 487 | 547 |
| vector-divergence-product | vector | 2 | 1 (1) | 0 | 30161 | 0/1 | yes | 611 | 398 |
