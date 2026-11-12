# Architecture

Minerva has three parts, kept separate so that mathematical correctness never depends on the
one most likely to be wrong.

```
 browser ──► web (Next.js) ──/api/v1/*──► API (FastAPI) ──► engine (SymPy, sandboxed processes)
    │                                         │  └────────► knowledge (YAML → SQLite FTS5)
    └──► site (static landing)                └───────────► PostgreSQL (users, sessions, history)
```

## A. Symbolic computation (`packages/engine`)

- **Parsing** (`latexparse.py`, `textparse.py`, `build.py`). SymPy's own LaTeX parser returned
  ambiguous trees for common inputs (for example `\sin^2 x + \cos^2 x`), and `sympify` uses `eval`.
  Minerva therefore parses LaTeX with a small recursive-descent grammar and text with Python's
  `ast` module under a strict whitelist, building SymPy objects through `Builder`, which also
  enforces size, depth, digit-count and exponent limits *before* SymPy evaluates anything.
- **Sandbox** (`sandbox.py`). Every operation runs in a child process from a pre-warmed
  `forkserver`, with a wall-clock timeout, `RLIMIT_CPU`/`RLIMIT_AS` (Linux), a concurrency
  semaphore and kill-on-timeout. Failures surface as `resource_limit`, never as guesses.
- **Operations** (`ops.py`, `service.py`):
  - `simplify`: SymPy `simplify`, then `verify_equal` (numeric sampling that respects assumptions,
    plus a symbolic zero test). Statuses: `ok`, `unchanged`, `not_verified`, `parse_error`,
    `resource_limit`, `internal_error`.
  - `series` (`series.py`): Taylor polynomial about a finite real point, order 1 to 12, from `sympy.series`.
    Shown only if both checks pass, computed from the original function with mpmath at 80 digits at random
    values of every other symbol: (1) the coefficients against a Cauchy-integral estimate (64-node trapezoid
    rule on circles; trusted at the first radius where the estimates at r and r/2 agree, so a pole inside a
    circle is never mistaken for agreement, tolerance per coefficient so a wrong high-order term cannot hide on
    a small circle), and (2) the remainder |f - P| falling at least as h^(order+1) along real (and, for a
    variable that may be complex, imaginary) directions. A pole, branch point, logarithm or `|x|` is
    `unsupported`; a failed leg is `not_verified` and shows no formula. The measured-error table is a
    measurement at a few distances for expressions with no other symbols, not a bound.
  - `differentiate` (`differentiate.py`): `sympy.diff` one order at a time, for one to four variables
    and a total order up to 10 (mixed partials included). Every step is compared with a 40-digit
    central finite difference of the previous expression at assumption-respecting random points;
    a complex variable is checked along the real and the imaginary direction, so `|z|`, `re`, `im`
    and conjugates come back `unsupported` ("declare the variable real") instead of as a formula
    that only looks like a derivative. There is no symbolic-zero leg: no independent symbolic test
    of a derivative exists, and the report says so. A derivative that fails the check is
    `not_verified` and never shown; one that grows past the size cap is `resource_limit`. Input
    expressions are whatever the parsers accept (no undefined functions `f(x)`, integrals or `d/dx`
    notation). A Dirac delta in the result (second derivative of `|x|`) carries a note.
  - `apply_rules`: for each candidate rule, match its left-hand side against every subexpression
    (with a `rest` wildcard for sums/products), reject matches that invent structure, check the
    rule's conditions against the declared assumptions, rewrite, and verify the rewrite.
  - `tensor_canonicalize`: `sympy.tensor.tensor` Butler-Portugal canonicalization (monoterm
    symmetries only).
  - `dimensions.py`: units as exact exponent vectors over the SI base units (base units, a fixed table of
    coherent derived units, named quantity kinds; no conversion) and dimensional consistency of equations
    (used by handbook checks and the `check_units` tool).
  - `pi_groups.py` (`dimensional_analysis`, `check_units`): dimensionless groups as the null space of the
    dimension matrix, each re-checked by rebuilding it as an expression and taking its dimension. With an
    unknown it reports `determined`, `up_to_function` or `missing`; for `missing` it gives the dimension the
    absent quantity must carry (verified by re-solving with it appended) and the handbook constants, alone or in
    pairs, that close the gap. It does not determine the function of the groups, dimensionless constants, or
    whether the list of quantities is complete.
  - `lagrangian.py` (`analyze_lagrangian`): `derive` builds the Euler–Lagrange equations, momenta and (for an
    invertible mass matrix) the accelerations with SymPy; `check_equations` then confirms every equation by
    nested numerical differentiation of the lambdified L along random cubic paths (`verify.euler_lagrange_check`,
    mpmath, no SymPy differentiation), and an unconfirmed equation makes the whole report `not_verified`.
    `find_conserved` finds cyclic coordinates, energy and rotations of coordinate pairs and verifies each on
    the equations of motion. With `quadratic_deadline` (the analyser only) it also runs `quadratic_integrals`: for
    L = (m/2)|q̇|² − V(q) in at most three Cartesian coordinates, integrals m K_ij(q) q̇_i q̇_j + U(q) with K a Killing
    tensor of flat space of degree ≤ 2 (exact linear algebra), the closed-form condition d_j U = 2 K_ij d_i V solved by
    a numerical null space at random points and parameters that is rationalised, the energy and the products of linear
    integrals removed, U built by Euler's formula on homogeneous pieces (or, in the degenerate degree −1 class that
    contains the Runge–Lenz vector, as a rationalised combination of monomial × potential term), and every candidate
    verified on the equations of motion. Bounded; "none found" proves nothing.
  - `expansion.py` (`expand_motion`): the accelerations from `derive`, expanded in the deviations from a
    user-given equilibrium (default 0; may use the parameters of L) by repeated differentiation in a scaling
    parameter, to degree 1-12. Shown only if the EL equations were re-confirmed (`check_equations`) and the
    solved accelerations satisfy them, the point is an equilibrium (accelerations vanish there: symbolic and
    numeric legs, refused otherwise), and each expansion passes `series.check_series` at random values of the
    deviations and parameters. Gives the linear matrices A, B (uddot = A u + B v), the characteristic polynomial
    (up to 4 coordinates) and, for at most 2 coordinates with B = 0, normal modes: ω² roots of det(A + ω² I)
    checked by substitution, eigenvectors checked by (A + ω² I) w = 0, the sign reported only when the declared
    assumptions decide it. The expansion is local; no stability claim is made.
  - `perturbation.py` (`perturbation_solve`): Poincaré–Lindstedt for one coordinate with an acceleration that
    depends on the position only, about a confirmed equilibrium (`expansion.prepare`). The small parameter is a
    named parameter of L (L must be a linear oscillator at 0; amplitude A symbolic) or the amplitude of the
    initial displacement. Each order is solved exactly on cosine polynomials with the strained time τ = ω t, ω_k
    removing the cos τ resonance. Shown only if the expansion passes `check_series`, the truncated solution
    leaves a residual in the unexpanded acceleration that falls at ε^(N+1) (mpmath, random parameters, ε halved
    twice), the initial conditions hold (symbolic and numeric) and every order is a finite cosine sum. The period
    ratio series is derived from ω and checked with `check_series`. Optional comparison with `integrate_motion`
    for user-given numbers, at ε and ε/2, reporting the observed order of the error; a measurement, omitted when
    the numerical run fails its own checks. Order 1-6.
  - `motion.py` (`integrate_motion`): numerical solution of those accelerations for given parameter values,
    initial conditions and time span, optionally for 2–6 values of one parameter. Pure-Python float adaptive
    Dormand–Prince 5(4) (relative tolerance 1e-12; no SciPy). A run keeps its trajectory only if (1) a fixed-step
    RK4 run with a step of a quarter of the primary's smallest accepted step in each segment agrees at every
    sample (1e-6, relative to each component's range), (2) integrating back from the end recovers the start
    (1e-6), and (3) every constant of motion verified by `find_conserved` drifts by less than 1e-6. Otherwise the
    run is `not_verified` and carries no data (chaotic or stiff motion over a long span, singularities);
    exceeding the step cap or the time budget is `resource_limit`. Parameters and initial values must respect
    the declared assumptions; a missing or unknown parameter is a `parse_error`; a singular mass matrix is
    `unsupported`. Stored series are rounded to 10 significant digits and capped at about 20 000 numbers.
    Not covered: field Lagrangians, generalised forces (damping, driving not derived from L), constraints,
    stiff solvers, sweeping initial conditions.
- **Extension point:** a future Cadabra or SageManifolds backend implements the same operations
  behind the engine interface, in its own process (see `docs/licensing.md`).

## B. Knowledge base (`packages/knowledge`)

- Source of truth: YAML (`data/chapters.yaml`, `data/entries/*.yaml`), validated by Pydantic models
  (`models.py`). Adding content is adding YAML.
- Entry kinds: identities (with `rule`), derivatives, integrals, series, symmetries, equations,
  constants, theorems. Each declares a `verification` level, and `checks.py` runs the matching
  checker for every entry in the test suite, so a wrong entry cannot be merged and an entry cannot
  claim more than it was checked for. `stated` entries are literature records, never applied.
- Retrieval (`kb.py`): a `Retriever` protocol with an `Fts5Retriever` (SQLite FTS5, BM25 over title,
  keywords, tags, text, plus structural feature tokens such as `f:log` or `op:square` derived from each
  rule's left-hand side). Semantic or structural-embedding retrievers can implement the same protocol.
  Retrieval only **proposes**; the engine decides.
- `identities.find_identities` orchestrates: features (sandbox) → candidates (FTS5) → apply (sandbox).
  The matcher (`engine/ops.py`) discards matches that invent roots, reciprocals or `I`, and matches that
  bind a variable used only as `n*w` (like the `a` in `sin(2*a)`) to a fraction of the input.
- Generated entries (`generate.py`): `data/entries/generated_*.yaml` are written by
  `python -m minerva_knowledge generate` from seeds in `generate.py`; edit the seed, never the YAML.
  A candidate is written only if it round-trips through the text parser, validates, is not a duplicate,
  passes its own check (`symbolic` only with a symbolic zero) and the engine applies it to its own
  left-hand side. The test suite fails when the committed files differ from a fresh generation.
- Benchmark (`benchmark.py`, `data/benchmark/identity_cases.yaml`): hand-written expression →
  expected/forbidden identities, run with `python -m minerva_knowledge benchmark`; results and limits in
  `docs/handbook-benchmark.md`.

## C. The conversational agent (`services/api/src/minerva_api/agent/`)

The main interface is a chat. The language model plans, asks clarifying questions, searches the handbook
and explains; it never computes and never decides correctness.

- **Loop** (`loop.py`): the conversation and the tool schemas go to the model (`llm.py`: an Ollama client
  that streams `/api/chat`); requested tool calls run in the sandbox through `tools.execute`, results go back to the
  model, until it answers. A step cap and a time cap end runaway loops.
- **Tools** (`tools.py`): `search_handbook`, `get_entry`, `simplify`, `differentiate`, `series`, `find_identities`, `tensor_canonicalize`, `dimensional_analysis`,
  `check_units`, `solve_polynomial`, `analyze_lagrangian`, `expand_motion`, `perturbation_solve` and `integrate_motion`. Each is a
  pure computation. `integrate_motion` reuses the engine's request model as its schema, and the prompt tells the model never to
  choose parameter values, initial conditions or time spans itself. The Lagrangian card has a form for them that
  posts to `POST /v1/chat/conversations/{id}/tools/integrate_motion`: the model is not involved (it works with
  `MINERVA_LLM_MODE=off`), the user's numbers go to the engine untouched, and the result is stored as an assistant
  message with the card and a trace step, so later turns see it. A tool returns a `Card` that the interface renders from the tool's own output, plus a compact
  text for the model. Bad or unknown calls become an error result the model can read, never an exception.
- **Honesty** (`segments.py`): after each turn, every formula in the model's prose is compared with the formulas
  that tool results and handbook entries of the conversation contain. Others are shown as *unchecked*.
- **Storage:** conversations and messages (per user, private); each assistant message stores its ordered blocks
  (prose and cards) and the tool trace so later turns have the same context.
- **Debug report:** each stored assistant message carries `trace` (tool name, the arguments the model sent, the text it
  read back, tool status and duration) and `debug` (`request_id`, model, mode, a fingerprint of the system prompt and
  tool schemas, per-call timings, and how each equation the model wrote was tested). With "Debug mode" on, the chat
  shows this and copies a Markdown report ending in a JSON block; `scripts/replay_trace.py` re-runs the recorded tool
  calls, and the `chat-trace-triage` skill says how to classify the result. This is shown only to the account that owns
  the conversation. The server log still holds no content: `chat_turn` adds `steps`, `tool_calls` and `tool_errors`
  counts, and the `request_id` in the report matches the `X-Request-ID` response header.
- **Models:** `MINERVA_LLM_MODE` is `off`, `ollama` (your own host: `MINERVA_LLM_BASE_URL`, `MINERVA_LLM_MODEL`) or
  `scripted` (deterministic stand-in for tests and UI work, development only). `scripts/eval_models.py` measures a
  model's tool-calling and honesty through the real loop; results in `docs/model-evaluation.md`.
- **Transport:** `POST /v1/chat/conversations/{id}/messages` streams server-sent events (`text`, `tool_call`, `card`,
  `message`, `error`, `done`).
- **Add image** (`vision.py`, `POST /v1/chat/formula-image`): the composer can read formulas from a picture. The
  browser shrinks and re-encodes it (dropping EXIF); the API checks size and magic bytes, and a vision model
  (`MINERVA_LLM_VISION_MODEL`, same Ollama host; empty hides the button) **transcribes** it to LaTeX and nothing
  more. `Engine.read_formulas` (sandbox) parses each candidate strictly and returns how Minerva reads it, without
  simplifying. The user confirms or edits in a panel and the formulas enter the draft as `$...$`; from then on they
  are ordinary user text. The picture is held in memory for the request only: never stored, never logged (the
  log line has status, duration and user id). Transcription quality depends on the model and is not scored yet.

## API (`services/api`)

FastAPI with OpenAPI docs at `/docs`. Auth: Argon2id passwords, random session tokens stored as HMAC
digests, `HttpOnly` `SameSite=Lax` cookie, same-origin check on unsafe methods, per-process rate
limits. Endpoints: `/v1/auth/*`, `/v1/compute/{simplify,differentiate,series,identities,tensor/canonicalize}`,
`/v1/reference/*` (public, read-only), `/v1/history`. History rows contain user mathematics and are
private to the account. Logging uses an allow-list of fields (no inputs, emails or tokens).

## Web (`apps/web`, `apps/site`, `packages/ui`)

- `apps/web`: Next.js 16 App Router. `src/proxy.ts` forwards `/api/v1/*` to the API at runtime
  (`MINERVA_API_URL`), redirects `/` to `/login` without a session cookie (convenience only, the API
  is the authority) and sets a nonce-based CSP. MathLive for input, KaTeX for output.
- `apps/site`: static export served by nginx, pre-rendered math, no cookies or analytics. Public URLs
  are build arguments (`NEXT_PUBLIC_APP_URL`, `NEXT_PUBLIC_REPO_URL`).
- `packages/ui`: theme tokens (light/dark) and the `Math` component.

## Deployment shape

Separate hostnames for the landing site and the app. The Helm chart in `deploy/helm/minerva` carries no
secrets or hostnames; they are supplied by values and repository secrets. Cluster, runner and
Cloudflare configuration live in separate infrastructure repositories.
