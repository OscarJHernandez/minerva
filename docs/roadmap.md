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
2. **More operations in the workspace:** integrals, limits, series, numeric evaluation, each with
   the same verification discipline (differentiate-back for integrals, interval/sample checks for limits).
   Derivatives are done (`differentiate`, any order up to 10, mixed partials, every step checked numerically).
   Later for derivatives: `\frac{d}{dx}` notation and undefined functions `f(x)` in the input, implicit
   differentiation, and mixed partials in the workbench form (the API and chat already take them).
3. **Tensor algebra in the UI:** declare tensors and symmetries, canonicalize, show unsupported
   multi-term identities explicitly.

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

- `LLMClient` interface with an Ollama-compatible endpoint configured by environment; feature off by
  default. Uses: suggest entry ids, explain a verified result in prose, normalise informal input to LaTeX for
  the user to confirm. Never computes or decides. Evaluate with a fixed prompt set; log nothing user-specific.

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
