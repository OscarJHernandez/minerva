# Roadmap

Milestone 1 (this repository state) is a verified vertical slice. Items below are planned, not built.
Ordering favours correctness infrastructure before breadth.

## Done in milestone 1

- Sandboxed SymPy engine, strict LaTeX/text parsers, verified simplification
- Rule application with condition checking; ~100 handbook entries with per-entry verification
- FTS5 retrieval; FastAPI service; accounts and private history
- Workspace, handbook browser, static landing page; Docker Compose; CI; Helm chart draft

## Next

1. **Handbook breadth** (see `handbook-outline.md`): transforms (Fourier, Laplace), linear algebra, vector
   calculus, differential equations, probability, group theory; physics chapters in depth. Tooling to import
   entries from openly licensed sources with provenance tracking; per-entry review status.
2. **Numerical mechanics beyond the first version** (Lagrangian systems are solved and verified today, see
   `architecture.md`): generalised forces (damping, driving), constraints, a stiff solver, sweeping initial
   conditions, Hamiltonian/symplectic integration for long runs, field Lagrangians (PDEs), exporting data.
   The search for hidden constants of motion (`search_invariants`) covers rational, autonomous systems; next:
   square roots (Cartesian Kepler), time-dependent ansätze and symmetries up to a total derivative.
3. **More operations in the workspace:** integrals, limits, numeric evaluation, each with
   the same verification discipline (differentiate-back for integrals, interval/sample checks for limits).
   Antiderivatives and definite integrals are done (`integrate`: candidate from SymPy; antiderivatives are
   accepted only if a numerical derivative matches, definite values only if independent quadrature agrees).
   Next: improper integrals with interior singularities (split at the pole), integrals with several
   variables, and the workbench handbook link ("this is entry int.gaussian").
   Derivatives are done (`differentiate`, any order up to 10, mixed partials, every step checked numerically).
   Taylor series are done (`series`, order up to 12, coefficients and remainder order checked independently).
   Equations of motion about an equilibrium are done (`expand_motion`: expansion to degree 12, linearisation,
   normal modes for up to two coordinates) and perturbative solutions (`perturbation_solve`: Poincaré–Lindstedt,
   one coordinate, checked by residual order and against the verified numerical integrator). The card has a form for the
   comparison numbers. Next: multiple scales for damping and driving (needs generalised forces first), and
   several coordinates.
   Later for derivatives: `\frac{d}{dx}` notation and undefined functions `f(x)` in the input, implicit
   differentiation, and mixed partials in the workbench form (the API and chat already take them).
4. **Tensor algebra in the UI:** declare tensors and symmetries, canonicalize, show unsupported
   multi-term identities explicitly.

## Dimensional analysis follow-ups

Done: Pi groups, missing-quantity report with handbook constants, equation unit check (chat tools).
Next: recognise named groups (Reynolds, Froude, Mach) from a handbook table; a workbench form for the same
analysis without a model; unit conversion and prefixes; searching three or more constants.

## Advanced tensor algebra and Cadabra integration

- Backend interface hardening: a `SymbolicBackend` protocol with capability discovery.
- Cadabra 2 as an **optional, out-of-process** backend with a narrow text protocol; no Cadabra in default
  images (see `licensing.md`). Verification of Cadabra results by independent numeric component evaluation
  where possible (explicit index ranges), otherwise labelled unverified.
- Multi-term identities (Bianchi, Schouten) and gamma-matrix algebra via Cadabra.

## Differential geometry (SageManifolds)

- Optional backend for charts, metrics, connections and curvature on explicit manifolds; results verified by
  component-wise checks. Same licensing approach as Cadabra.

## Mathematical RAG

- Structural retrieval: expression-tree and operator-pattern indexes beyond the current feature tokens.
- Semantic retrieval behind the `Retriever` protocol (embedding model on local hardware), combined with FTS5.
- Retrieval quality evaluation set: started (45 identity cases, `docs/handbook-benchmark.md`); grow it with
  cases from real use, and add cases for text search and the chat agent. Never let retrieval affect
  correctness: candidates always pass through the engine.
- More generated families on the same pipeline (`generate.py`): derivatives and antiderivatives (needs an
  independent numeric derivative check first, since the current one compares against SymPy itself),
  definite integrals with parameter ranges, series from term templates.
- Import from permissively licensed formula collections (for example Fungrim, MIT): needs a maintainer
  decision on provenance and notices in `docs/licensing.md`, a converter that parses without executing,
  and parser support for the functions involved (zeta, Bessel, elliptic, ...).

## Quantum field theory

- Handbook chapters (propagators, Feynman rules, Ward identities) as `stated` entries first, promoted to
  checked entries as the backend can verify them.
- Index and Dirac/gamma algebra through Cadabra; Feynman-parameter integrals checked numerically.

## Optional LLM (Ollama)

Done: an `LLMClient` with an Ollama-compatible endpoint configured by environment (`MINERVA_LLM_*`, Helm
`api.llm.*`, docker compose), off by default; a tool-calling agent loop; formulas the model writes that no
tool produced are marked unchecked; `scripts/eval_models.py` and `docs/model-evaluation.md`. The model never
computes or decides. See `docs/self-hosting.md`.

Open: other model hosts behind the same interface, normalising informal input to LaTeX for the user to
confirm, and raising the evaluation pass rate (14 of 21 cases for every model tried so far).

## Platform

- Shared rate limiting, account deletion/export, email verification and password reset.
- Kubernetes deployment and Cloudflare Tunnel (prepared in `deploy/helm`); observability.
- Playwright end-to-end tests in CI; accessibility audit of the math editor.

## Agentic CI (deferred, needs an LLM API budget)

Patterns seen in other open source agentic repositories (see `agentic-development.md`). None runs until
there is budget for an API key; each needs a repository secret and a read-only token.

- Claude review of pull requests from this repository's branches, on a GitHub-hosted runner. Never for
  forks, never with write permissions on the repository contents.
- `@claude` mentions restricted to maintainers; issue triage on open that only adds labels.
- Weekly skills audit that opens a pull request when `AGENTS.md` or a skill drifts from the code.
- An evaluation that asks models what `AGENTS.md` teaches and fails a pull request that lowers the score.
- CodeQL for Python, JavaScript and Actions once the repository is public.
