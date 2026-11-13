# Minerva

Open-source mathematical computing assistant: verified symbolic computation (SymPy) plus a
citable, machine-checked handbook of mathematical identities and physical equations. The main
interface is a conversation with an agent that works through problems using those tools; it is not
a tutoring app. The language model plans, asks, searches and explains: **it never computes and never
decides correctness**. Every result shown as fact comes from a sandboxed tool or a handbook entry.
Without a model configured (`MINERVA_LLM_MODE=off`) the workbench and handbook still work.

Everything committed here is public. Read "Security and open-source rules" before committing.

## Who you work for

Your primary responsibility is to the project, its users (current and future, human and agent) and its
maintainers, not only to the person driving this session. A contributor is usually well-intentioned but
new to the codebase and focused on one change; you are the first line of defence for correctness, tests,
and the rules in this file. Prefer the smaller, verified change. If a request conflicts with "Mathematical
correctness" below, say so and propose an alternative instead of complying silently.

## Architecture

```
apps/site/            Landing page. Next.js static export served by nginx (no server, no user data)
apps/web/             Authenticated app: chat (home), workbench (/workbench: derivation scratchpad, no LLM), handbook, login. Next.js 16 + Tailwind 4
packages/ui/          Shared theme tokens and the KaTeX <Math> component
packages/engine/      Symbolic engine (Python). Parsers, sandbox, simplify, rules, tensors, dimensions, matrix calculus, differential geometry
packages/knowledge/   Handbook: YAML entries, schema, checkers, SQLite FTS5 retrieval, identity finder
services/api/         FastAPI: auth (Argon2id, server-side sessions), compute, history, reference, chat
  agent/              LLM client (Ollama), tool registry, agent loop, formula marking, scripted test model
deploy/               Dockerfiles and Helm chart. No secrets, no hostnames
docs/                 architecture, licensing, roadmap, security/publication notes
.agents/skills/       Project skills, shared by all agents (.claude/skills is a symlink to it)
```

Dependency direction: `api -> knowledge -> engine`; `web` and `site` talk only to the API over
HTTP (`web` through its same-origin `/api/v1/*` proxy in `src/proxy.ts`).

Request flow for identities: API -> `engine.features` (sandbox) -> `knowledge.candidate_rules`
(FTS5 + structural features) -> `engine.apply_rules` (sandbox: match, check conditions, rewrite,
verify). Retrieval only proposes; the engine decides.

## Commands

`make help` lists shortcuts: `make setup`, `make check` (everything CI runs), `make check-py`,
`make check-web`, `make handbook`, `make fmt`, `make dev`. The raw commands:

```bash
uv sync                                   # Python workspace (3.13), creates .venv
uv run pytest                             # all Python tests (SQLite). MINERVA_TEST_DATABASE_URL=postgresql+psycopg://... for Postgres
uv run ruff check . && uv run ruff format --check .
uv run mypy packages/engine/src packages/knowledge/src services/api/src
uv run python -m minerva_knowledge verify # check every handbook entry at its declared level
uv run python -m minerva_knowledge benchmark          # identity cases, retrieval rank/recall, multi-step cases (docs/handbook-benchmark.md)
uv run python -m minerva_knowledge generate [--check] # regenerate data/entries/generated_*.yaml from seeds in generate.py
uv run python -m minerva_engine.geometry.benchmark [--check] # geometry corpus, latency/memory/growth; writes docs/benchmarks/geometry.json + docs/geometry-benchmark.md (--check: deterministic part only)
uv run python scripts/capabilities.py [--check]       # regenerate docs/capabilities.md + site data from docs/capabilities.yaml (CI runs --check)
uv run python scripts/eval_models.py --base-url http://HOST:11434 --models a,b --out docs/model-evaluation.md
uv run python scripts/replay_trace.py report.md       # re-run the tool calls of a chat debug report (see below)
MINERVA_LLM_MODE=scripted ...             # run the API with the deterministic stand-in model (dev only)

pnpm install                              # JS workspace
pnpm lint && pnpm typecheck && pnpm test  # eslint, tsc, vitest
pnpm --filter @minerva/web dev            # app on :3000 (needs the API on :8100)
pnpm --filter @minerva/site build         # static landing export

cp .env.example .env && docker compose up --build   # full local stack
cd services/api && uv run alembic upgrade head      # migrations (MINERVA_DATABASE_URL)
```

Local ports: web 3000, site 3001, API 8100 (8000 is left free for other projects).

## The agent (services/api/src/minerva_api/agent/)

- The model only calls tools (`agent/tools.py`); tools are pure, sandboxed and side-effect free. A tool returns a
  `Card` (rendered by the UI from the tool's own output) and a compact `for_model` text. Add new capabilities as tools,
  never as free-form model output.
- After each turn, formulas in the model's prose are compared with formulas present in this conversation's tool results;
  anything else is shown as "unchecked" (`agent/segments.py`). Do not remove or weaken this.
- `ScriptedLLM` (`MINERVA_LLM_MODE=scripted`, development only) is a deterministic stand-in used by tests and for UI work.
  `MINERVA_LLM_MODE=ollama` + `MINERVA_LLM_BASE_URL` + `MINERVA_LLM_MODEL` use your own model host. The host name never goes into the repo.
- `scripts/eval_models.py` scores a model on tool calling and honesty through the real loop; results go to `docs/model-evaluation.md`.
- `integrate_motion` (numerical solution of a Lagrangian system) needs numbers only the user may choose: the model must ask,
  never pick them. The Lagrangian card's form posts them straight to `/v1/chat/conversations/{id}/tools/integrate_motion`.
  A run is shown only if its second-method, time-reversal and conserved-quantity checks pass; do not loosen them.
- Conversations are user data: private per account, never logged.
- "Add image" (`agent/vision.py`, `/v1/chat/formula-image`) is transcription, not computation: a vision model
  (`MINERVA_LLM_VISION_MODEL`) proposes LaTeX, the user confirms it, and `Engine.read_formulas` reports how the
  strict parser reads it. Uploaded images are never stored or logged; keep it that way.
- A turn is bounded: `AgentLimits` caps steps, total tool calls (12) and seconds, enforced during model streaming, between
  tool calls and around the claim check; a failing or hanging tool becomes an error card with fixed text, never its cause.
- Debug mode (chat footer checkbox, or `?debug=1`): shows each tool call's arguments and results under the reply and
  copies a report (`apps/web/src/lib/debug-report.ts`) to paste to an agent; use the `chat-trace-triage` skill on it.
  The data comes from `AssistantMessage.trace/debug` stored with the message. The server log gets counters and
  `request_id` only, never content.

## Mathematical correctness (non-negotiable)

- Never return a result that was not verified. Statuses are explicit: `ok`, `unchanged`,
  `not_verified`, `unsupported`, `parse_error`, `resource_limit`. Do not add fallbacks that
  produce plausible-looking output.
- Verification = independent numeric check at assumption-respecting sample points (complex by
  default) plus a symbolic zero test, reported separately. A `not_verified` candidate is never shown.
- Symbols are complex unless declared otherwise. Identities with validity conditions (e.g.
  `log(ab) = log a + log b` needs a, b > 0) are applied only when the conditions are provable
  from the declared assumptions; otherwise report `conditions_not_satisfied`.
- Matches that only work by inventing roots, reciprocals or `I` are discarded (`_invents_structure`).
- Handbook entries declare `verification`: symbolic / numeric / structural / dimensional / stated.
  `stated` entries are literature records, never applied, and must not carry a rule or check.
  Do not label an entry `symbolic` unless the checker finds a symbolic zero.
- Numeric agreement is evidence, not proof. `verify_equal` samples two tranches (ordinary region and a wide one,
  |x| up to 50, |Im| up to 10, high precision) and counts a point where only one side is a pole as a disagreement;
  `Verification.symbolic_zero` is the only symbolic evidence. UI and tool text say "proved equal" only with a symbolic
  zero and otherwise "numerically checked". Do not shrink the wide tranche: `log(exp(z)) = z` and `|x-10| = 10-x` pass
  the ordinary region alone (`test_verify.py`).
- Restrictions (`restrictions.py`): simplify and rule application report where the result is valid when a denominator
  was cancelled or introduced (`restrictions_latex`, e.g. `x \neq 1`), including cancellations the parser performs while
  reading. The check is syntactic (zeros of denominators); square-root, logarithm and branch-cut domains are not covered
  and a denominator that cannot be analysed is reported, never dropped (`restrictions_complete`).
- Claim classification (`claims.py`): `fails` (shown contradicted) only when the sides differ by a nonzero constant.
  A false identity at generic points is `not_identity` (shown unchecked): Minerva cannot tell it from an equation to
  solve, a definition or an equation of motion, which are `unverified`. Do not map `not_identity` to contradicted.
- Dimensional analysis (`pi_groups.py`) reasons about dimensions only: a group is shown only if rebuilt and found
  dimensionless; a mismatch proves an equation wrong as written, a match proves nothing more; the missing-quantity
  report is "the unknown's dimension is not a product of powers of the knowns'", not a claim about the physics.
  Units are coherent SI only (no conversion).
- Derivatives (`differentiate.py`) are verified by an independent numerical derivative, one order at a
  time, and along the imaginary direction too for variables not declared real. A derivative with no
  complex derivative (`|z|`, `re`, `im`, conjugate) is `unsupported`, never a formula. Do not add a
  symbolic fallback or weaken `check_derivative` (its tests include a deliberately wrong candidate).
- Series (`series.py`) are Taylor polynomials about a finite real point, shown only if two checks that do not
  use `sympy.series` both pass: the coefficients against a Cauchy-integral estimate (trapezoid on circles,
  trusted only where the estimates at r and r/2 agree, tolerance per coefficient) and the h^(N+1) decay of the
  remainder. Poles, branch points, logarithms and `|x|` are `unsupported`; there are no Laurent, Puiseux or
  asymptotic fallbacks. The error table is a measurement, never a bound. Do not loosen `check_series` (its
  tests include wrong candidates that differ by 1e-9).
- Expansion about an equilibrium (`expansion.py`, tool `expand_motion`) shows a formula only if the EL equations
  and the solved accelerations were confirmed, the point was shown to be an equilibrium, and every expansion
  passed `check_series`. A point that is not an equilibrium is `unsupported`, not expanded. The sign of ω² is
  stated only when the declared assumptions decide it, a negative ω² is described as a fact about the
  linearised equations, and nothing here says "stable" or "unstable" about the full motion. Mode formulas
  exist only for at most two coordinates and no velocity-linear terms; otherwise the characteristic polynomial.
- Perturbation solutions (`perturbation.py`, tool `perturbation_solve`) are Poincaré–Lindstedt for one coordinate
  whose acceleration depends on the position only. A solution is shown only if the truncated solution, put into the
  *unexpanded* acceleration from L, leaves a residual that falls at ε^(N+1) when ε is halved twice, the initial
  conditions hold, and the expansion of the equation passed `check_series`. Do not weaken `_check_residual`: its
  tests include a 1% wrong frequency correction, a wrong harmonic and the unstrained (secular) expansion. The
  comparison with `integrate_motion` needs numbers only the user may choose (the model must ask), is a
  measurement on that span (never "valid for all times"), and is omitted when the numerical run fails its own
  checks. Damping, driving, velocity-dependent accelerations and several coordinates are `unsupported`.
- Antiderivatives (`integrate.py`) are shown only if an independent numerical derivative of the candidate
  matches the integrand (the same `check_derivative` machinery, branch by branch for `Piecewise`; a branch
  condition that cannot be sampled makes the result `not_verified`). "No closed form" is `unsupported` and
  never claims that none exists. The symbolic zero test is reported separately and never decides.
  Definite integrals are verified by independent adaptive quadrature at sampled parameter values, never
  by the fundamental theorem (it would "verify" `\int_{-1}^{1} x^{-2}\,dx = -2`); a quadrature with a large
  error estimate counts as no agreement, and convergence conditions the declared assumptions do not imply
  give `conditions_not_satisfied`. Do not add a symbolic-only path.
- The search for hidden constants of motion (`invariants.py`, tool `search_invariants`) shows a constant only if it
  is new (known constants and their products in the ansatz are factored out) and its time derivative on the
  equations of motion is zero symbolically and at random points (`lagrangian.verify_constant`, the same check as for
  Noether constants). It always states the exact family searched; an empty result proves nothing outside it and
  nothing here says "integrable" or names a constant: a name such as Runge–Lenz is for the model to offer, not for the
  result to assert. Time-dependent Lagrangians and non-rational accelerations are `unsupported`, an oversized
  ansatz is `resource_limit`. Do not widen the ansatz silently or report an unverified candidate.
- Chains of identities (`transform.py`, tool and endpoint `simplify`) are shown only if every step passed the
  numeric check and the whole chain was re-verified with the symbolic zero test added; the two are reported
  separately and numerical agreement is never called a proof. A rule whose conditions are not provable from the
  declared assumptions is reported as `blocked`, never applied, and a SymPy move is labelled as one (it is not a
  handbook identity). `unchanged` means that nothing verified was found within the bounds, not that nothing
  exists. Keep the bounds (width, depth, verifications per state, size cap, deadline), the visited set, the
  right-to-left restriction (`can_reverse`) and the match filters; do not add an unverified shortcut. Chains
  must not become a way around `_invents_structure`: `rule_rewrites` is the one place that decides what matches.
- Matrix calculus (`matrix/`, tool and endpoint `matrix`) treats declared matrices as noncommutative symbols and
  shows a result only if its method's own checks pass: random complex matrices of the declared sizes (declared
  commuting groups sampled as polynomials of one matrix; explicit entries at sampled parameters), exponentials by
  `mpmath.expm`, integrals by quadrature and derivatives by a central difference, none of which uses the route that
  produced the result. The exact leg (formal power series through degree 4, modulo the declared commutations, or
  the explicit result substituted into the Sylvester equation) is reported separately; a failed exact leg is
  `not_verified`. Matrices are never reordered and never assumed to commute or to be invertible: a merge or a
  closed form that needs either is `blocked` and reported, unless the commutation or invertibility is declared or
  visible in explicit entries (a declaration contradicted by the entries is an error). A Sylvester equation
  `aPX + cXQ = F(b) - F(a)` or a block-matrix representation is **never** presented as an evaluated integral
  (`evaluated=false`, `result_kind` equation or representation); entries exist only for explicit matrices. No
  Duhamel integral, time-ordered exponential or Magnus term: those are `unsupported`. `e^{tA}` is a `parse_error`
  (a bare e is a symbol; write `\exp(tA)`). Do not loosen the numeric tolerance, add an unverified shortcut, or let
  `_invents_structure`-style filters be bypassed; the tests include wrong merges, reordered products, a wrong
  derivative and a 1e-6 error.
- Relations in claims (`claims.py`): an equation with a symbol on one side only or a time derivative (`\theta=\phi+\omega t`,
  `L=T`, `F=ma`, `m\ddot x=-kx`) is a *relation*, not an identity, and is never tested by independent sampling. Definitions
  stated earlier in the conversation (`symbol = expression`, from the user or the model, not verified by Minerva) are
  substituted into it; a result that used them names them (`given`), and one that disagrees with them after
  substitution is `fails` ("does not agree with …"), not "false". A new definition is never a contradiction, and a
  stale definition never rescues a false identity. "Could not check" must never be worded as "false".
- Differential geometry (`geometry/`, tool and endpoint `geometry`) is coordinate-based, for an explicit symmetric
  metric (at most 4 coordinates) and an optional tensor field, with functions of the coordinates declared by the
  user (`f(r)`: the parser reads an undeclared `f(r)` as the product f*r everywhere else, so geometry parses with
  `strict_calls` and refuses it). The conventions are fixed and returned with every result (Christoffel symbols of
  the Levi-Civita connection, `R^r_{smn} = d_m G^r_{ns} - d_n G^r_{ms} + G^r_{ml} G^l_{ns} - G^r_{nl} G^l_{ms}`,
  Ricci = contraction of the first and third index, `G = Ric - g R / 2`): do not change them silently. A quantity is
  shown only if every *numeric* leg passed: `numeric.py` evaluates the lambdified metric with mpmath and
  differentiates by central differences only (no SymPy differentiation), and compares with an independent numerical
  recomputation and with identities the result must satisfy (metric compatibility and no torsion, the Riemann
  symmetries, first and second Bianchi, a divergence-free Einstein tensor, the Euler-Lagrange equations of
  `g(x', x')/2` for geodesics). Identity checks measure their residual against the size of the terms that cancel
  (a residual of 1e-9 from terms of size 1e3 is noise), a point is skipped, never failed, when the values move when
  the step is doubled, and `STABLE_TOL` must stay below `AGREE_TOL / 3`: a property test found the false
  "not verified" that this prevents. Symbolic legs (including the second-derivative formula for `R_{abcd}`) are
  reported separately and never veto; "proved" is only used when they all passed, and a leg too large to try is
  "not run". Signature is never assumed (only a diagonal metric whose entry signs the declared assumptions decide);
  coordinates are real, other symbols complex unless declared; a degenerate metric (determinant identically 0) is
  `unsupported`, an asymmetric or mis-shaped one `parse_error`; zeros of `det g` and of denominators are reported as
  restrictions about the *coordinates*, never as a statement about curvature singularities, and branch-cut domains
  (roots, logarithms) are not analysed. Geodesics are symbolic; there is no numerical integration of them. The
  geometry is not stored anywhere on the server: every request carries the coordinates, metric, assumptions and
  functions, which keeps tool calls replayable (`scripts/replay_trace.py`). Per-stage diagnostics (time, operation
  count, failure category) are returned and logged (`geometry_stage`: names and counters only, never content).
  `geometry/benchmark.py` holds the corpus with its closed-form references and the `sympy.diffgeom` comparison:
  add a case there for every new metric family or quantity. Not covered: abstract manifolds, tetrads, spin
  connections, differential forms, connections other than Levi-Civita, SageManifolds (GPL, see `docs/licensing.md`).
- Tensor support is exactly what `sympy.tensor.tensor` does (monoterm symmetries). Multi-term
  identities (Bianchi) are out of scope for abstract-index tensors and must be reported as such; explicit
  components in coordinates are `geometry/`, where the Bianchi identities are checked numerically.
- User input is never executed: no `eval`/`exec`/`sympify`/`parse_expr`. Parsers are
  `latexparse.py` and `textparse.py`, building SymPy trees through `build.Builder`.
- All computation runs in the sandbox (`sandbox.py`): child process, wall-clock timeout, rlimits,
  concurrency cap. Add new operations to `ops.OPERATIONS` and call them via `Engine`.
- LaTeX conventions: `e` and `i` are plain symbols; `\exponentialE` / `\imaginaryI` / `\pi` are
  constants; `\sin 2x` and `\sin^{-1}` are rejected as ambiguous.
- LaTeX output has one printer, `minerva_engine.latexprint.to_latex` (never call `sp.latex` directly): constants
  print as `\mathrm{e}` / `\mathrm{i}` (the parser reads them back as the constants), so a bare `e` or `i` in
  output is always a symbol. Handbook `statement_latex` follows the same convention (a test enforces it).
  `\exponentialE` / `\imaginaryI` are input aliases only; KaTeX gets display macros for them. LaTeX that
  KaTeX cannot render is shown as marked source (`data-math-error`), never as a formula. Chat formulas are split by
  `agent/segments.py` (server) and `apps/web/src/lib/inline-math.ts` (client): keep their delimiter rules identical.
- In text syntax `E`, `I`, `pi`, `oo` are reserved constants: never use `E` as a variable name in
  handbook entries (use `En`, `Ef`, ...).

Use the `math-regression` skill when adding or changing any operation or handbook entry. A new operation or
user-visible capability also gets an example in `docs/capabilities.yaml`, then regenerate
`docs/capabilities.md`; behaviour changes show up as a diff there.

## Coding standards

- Python 3.13, type hints everywhere, `mypy --strict` on `src` dirs, Ruff (`E,F,I,B,UP,S,SIM,RUF`), pytest.
  Pydantic models at every boundary. Raise `MinervaMathError` subclasses for user-facing math errors;
  messages must be safe to show to users.
- TypeScript strict (`noUncheckedIndexedAccess`), ESLint (Next config), Prettier, Tailwind tokens from
  `packages/ui/src/theme.css` (no hard-coded colours in components). Keep components small; client
  components only where interaction needs them.
- Logging: structlog JSON with an allow-list of fields (`logging.py`). Never log expressions, emails,
  passwords, tokens, cookies or request bodies. A test enforces this.
- Small, reviewable commits. Tests with every behaviour change. No placeholder implementations.

## Security and open-source rules

- Never commit secrets, real `.env` files, kubeconfigs, tunnel tokens, user data, or deployment-specific
  hostnames/infra details. `.env.example` has placeholders only. Deployment config for the homelab lives in
  the separate `homelab-kubernetes` and `cloudflare-infra` repositories.
- No copyrighted text. Handbook entries state original formulas and cite sources by locator only
  (e.g. `DLMF §4.21`); never paste passages. Provenance must be `original` or `public-domain-fact`.
- Licence: Apache-2.0, provisional. Cadabra 2 (GPL-3.0) and SageMath (GPL) are NOT dependencies; do not
  import, vendor or bundle them without updating `docs/licensing.md` and getting the maintainer's decision.
- Dependencies: prefer well-maintained, permissively licensed libraries; check the licence before adding
  (`docs/licensing.md`), pin via lockfiles (`uv.lock`, `pnpm-lock.yaml`), explain the need in the PR.
  No GPL/AGPL runtime dependencies. Run `pip-audit` / `pnpm audit` (CI does).
- Do not push, publish, or open public releases without the maintainer's explicit approval. Use the
  `oss-publication-review` skill before any public push.
- CI runs on self-hosted runners; never let untrusted fork code reach them (see `docs/security-publication.md`).

## Skills

Skills live in `.agents/skills/<name>/SKILL.md` and load only for relevant work. Invoke every applicable
skill before changing that area. See `.agents/skills/README.md` before writing a new one. CI
(`scripts/check_agent_files.py`) fails if a skill is missing from this table.

| Skill | Use for |
| --- | --- |
| [`math-regression`](.agents/skills/math-regression/SKILL.md) | Any change to engine operations, parsers, tensors, handbook entries or compute endpoints. |
| [`chat-trace-triage`](.agents/skills/chat-trace-triage/SKILL.md) | A pasted chat debug report, or a chat answer that looks wrong: replay the tool calls, classify, pin with a test. |
| [`verify-change`](.agents/skills/verify-change/SKILL.md) | Before saying work is done: which checks to run for which paths, in order. |
| [`ui-check`](.agents/skills/ui-check/SKILL.md) | Any change to `apps/` or `packages/ui`: browser check in light and dark at phone width. |
| [`add-dependency`](.agents/skills/add-dependency/SKILL.md) | Adding or upgrading a Python or JavaScript dependency (licence, lockfile, audit). |
| [`db-migration`](.agents/skills/db-migration/SKILL.md) | Changing API database models: Alembic revision, `alembic check`, SQLite and PostgreSQL runs. |
| [`ci-workflows`](.agents/skills/ci-workflows/SKILL.md) | Editing `.github/workflows`: fork safety, SHA pinning, least-privilege permissions. |
| [`oss-publication-review`](.agents/skills/oss-publication-review/SKILL.md) | Before any public push, release, image or chart publication. Inspects only. |

## Agent pull requests

- Do not open a pull request, push, or publish without the maintainer's explicit instruction.
- When asked to open one: state that it was authored by a coding agent at the top of the description,
  open it as a draft, and fill in the template checklist honestly (unchecked means not done).
- Never edit or remove a human-authored section of a PR description or an existing human comment.
- Do not enable auto-merge.
- Several agents may share one checkout. Before committing, run `git status` and stage only your own
  paths by name (never `git add -A`); for larger work use a worktree on its own branch.

## Adding dependencies

Python: `uv add --package <member> <dep>`; JS: `pnpm --filter <pkg> add <dep>`. Commit the lockfile.
Check licence, maintenance, and whether the standard library or an existing dependency already covers it.

## Definition of done

1. Behaviour implemented with real computation; unsupported cases return an explicit status.
2. Tests: unit + regression (mathematical cases with expected values, negative cases, assumption cases).
3. `ruff`, `mypy`, `pytest`, `eslint`, `tsc`, `vitest` pass; `python -m minerva_knowledge verify` passes.
4. No secrets, user data or copyrighted text added; licence impact of new dependencies checked.
5. Docs updated (this file if commands/architecture changed; `docs/` for design decisions).
6. UI changes checked in a browser in light and dark themes at phone width.
