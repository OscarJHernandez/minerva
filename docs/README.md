# Documentation

| Document | What it is for |
| --- | --- |
| [capabilities.md](capabilities.md) | What Minerva can do today, with examples run through the real engine (generated from [capabilities.yaml](capabilities.yaml)) |
| [openrouter-setup.md](openrouter-setup.md) | Use OpenRouter cloud models: in-app personal key, or a deployment-wide default on Kubernetes |
| [self-hosting.md](self-hosting.md) | Build the Docker images, run your own instance, connect your own Ollama model |
| [architecture.md](architecture.md) | How the engine, handbook, API, agent and web app fit together |
| [roadmap.md](roadmap.md) | What is planned, in order |
| [observability.md](observability.md) | Traces, structured events, debug reports, replay, the agent benchmark, troubleshooting and what is never logged |
| [reference-benchmark.md](reference-benchmark.md) | Problems with answers established outside Minerva: matrix exponentials, delta integrals, substitutions, mechanics |
| [model-evaluation.md](model-evaluation.md) | How candidate language models score on tool calling and honesty |
| [conserved-quantities.md](conserved-quantities.md) | How conserved quantities are discovered, what is and is not established, and the generated-Hamiltonian benchmarks |
| [invariant-benchmark.md](invariant-benchmark.md) | Adaptive against fixed-parameter search on generated Hamiltonians (generated) |
| [handbook-benchmark.md](handbook-benchmark.md) | Identity retrieval quality on hand-written cases |
| [transformations-benchmark.md](transformations-benchmark.md) | Substitute, collect, solve for a variable and change one part: baseline of the older operations, results and latency |
| [variational-calculus.md](variational-calculus.md) | Functionals of several functions: first and second variation, boundary terms, numerical action, stationary paths; conventions, limits and how results are checked |
| [variational-benchmark.md](variational-benchmark.md) | 21 closed-form reference cases (generated; data in [benchmarks/variational.json](benchmarks/variational.json)) |
| [statistical-mechanics.md](statistical-mechanics.md) | Partition functions and thermodynamic derivations: systems, held-fixed derivatives, how each result is checked, what is never assumed, what is not supported |
| [statmech-benchmark.md](statmech-benchmark.md) | Statistical mechanics against closed forms, and refusals with their reasons (generated; data in [benchmarks/statmech.json](benchmarks/statmech.json)) |
| [geometry-benchmark.md](geometry-benchmark.md) | Differential geometry: correctness against closed forms and `sympy.diffgeom`, latency, expression growth and memory (generated; data in [benchmarks/geometry.json](benchmarks/geometry.json)) |
| [matrix-benchmark.md](matrix-benchmark.md) | Matrix calculus: coverage, method selection, verification and timing on hand-written cases, with a baseline |
| [statistics-benchmark.md](statistics-benchmark.md) | Statistical derivations (sums, expectations, families, maximum likelihood, Fisher information, Bayesian updates, matrix gradients): hand-derived references, adversarial cases, a baseline and a regression gate |
| [creative-telescoping.md](creative-telescoping.md) | Integrals with a parameter: discovering and proving L f = dG/dx, convergence and boundary terms from declared assumptions, the equation and its checked solution; supported class, limits, references |
| [creative-telescoping-benchmark.md](creative-telescoping-benchmark.md) | 29 problems against independent references: discovery, operator order, exact proof, failure classes (generated; data in [benchmarks/creative_telescoping.json](benchmarks/creative_telescoping.json)) |
| [integration-strategies.md](integration-strategies.md) | Deterministic integration techniques behind `integrate`: detection, verified transformations, what is blocked and never applied |
| [integration-strategies-benchmark.md](integration-strategies-benchmark.md) | Each technique against quadrature, and the default call against the requested strategy (generated; data in [benchmarks/integration_strategies.json](benchmarks/integration_strategies.json)) |
| [handbook-outline.md](handbook-outline.md) | Planned handbook chapters |
| [licensing.md](licensing.md) | Licence assessment, including the GPL-licensed integrations |
| [security-publication.md](security-publication.md) | Security and publication notes |
| [agentic-development.md](agentic-development.md) | How coding agents work in this repository |

The same capability record is published on the landing page's Docs page.
