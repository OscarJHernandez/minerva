# Offline orchestration suite

The real agent loop, tools and claim check with a replayed model (no network). Tokens are estimates (`budget.estimate_tokens`, deliberately pessimistic); timings are one run on one machine.

16/16 trajectories end mathematically correct · 36 model calls · 21 tool calls (2 invalid, 1 unknown names) · ≈422240 tokens in, ≈1228 out · 6 unbacked claims

| Trajectory | Category | Model calls | Tool calls (ok) | Routed | Tokens in | Evidence | Math ok | Tool ms | Claim ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pendulum-original-failure | notation | 4 | 3 (2) | 0 | 59866 | 4/4 | yes | 3593 | 943 |
| pendulum-routed-latex | lagrangian | 1 | 1 (1) | 1 | 14923 | 4/4 | yes | 0 | 956 |
| pendulum-routed-text-velocities | notation | 1 | 1 (1) | 1 | 14913 | 4/4 | yes | 0 | 968 |
| pendulum-model-calls-with-text | notation | 2 | 1 (1) | 0 | 28990 | 4/4 | yes | 921 | 947 |
| pendulum-assignment-and-greek | notation | 2 | 1 (1) | 0 | 28991 | 4/4 | yes | 891 | 943 |
| pendulum-wrong-sign-stays-unbacked | lagrangian | 2 | 1 (1) | 0 | 29004 | 3/4 | yes | 884 | 1490 |
| cyclic-oscillator | conserved | 2 | 1 (1) | 0 | 28672 | 1/1 | yes | 456 | 456 |
| simplify-with-functions-prefix | algebra | 2 | 1 (1) | 0 | 17214 | 1/2 | yes | 989 | 409 |
| simplify-text-syntax-then-fix | notation | 3 | 2 (1) | 0 | 25914 | 1/1 | yes | 2277 | - |
| unreadable-arguments-then-valid | recovery | 3 | 2 (1) | 0 | 25791 | 1/2 | yes | 991 | 393 |
| tool-call-written-as-prose | recovery | 3 | 1 (1) | 0 | 25698 | 1/1 | yes | 584 | - |
| unknown-tool-is-refused | recovery | 3 | 2 (1) | 0 | 26042 | 1/1 | yes | 579 | - |
| integrate-antiderivative | calculus | 2 | 1 (1) | 0 | 17101 | 1/1 | yes | 574 | - |
| matrix-expansion-noncommuting | matrix | 2 | 1 (1) | 0 | 21923 | 0/2 | yes | 407 | 420 |
| ode-first-order | ode | 2 | 1 (1) | 0 | 27037 | 0/1 | yes | 544 | 557 |
| vector-divergence-product | vector | 2 | 1 (1) | 0 | 30161 | 0/1 | yes | 592 | 396 |
