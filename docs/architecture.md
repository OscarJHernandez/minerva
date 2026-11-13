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
    `resource_limit`, `internal_error`. The API and the chat tool no longer call it directly: they call
    `transform` (below), which includes it as one of its moves.
  - `transform` (`transform.py`, `Engine.transform`): a bounded search for a chain of rewrites toward a goal
    (`simplify`: fewest operations; `expand`; `factor`; `rewrite` in a given set of functions). Moves are the
    handbook rules proposed by retrieval (each at every place it matches, from `ops.rule_rewrites`; read right
    to left only if the right side binds every variable and is not a bare number or symbol) and SymPy's
    `simplify`/`expand`/`factor`/`cancel`, labelled as SymPy moves. A move enters the search only after
    `verify_equal(..., symbolic=False)` agrees at assumption-respecting points; rules whose conditions are not
    provable are reported (`blocked`), never applied. Beam width 4 (cap 6), depth 4 (cap 6), 8 verifications per
    state, a visited set, a size cap on intermediate results, a stop after two levels without a better cost, and
    the 0.7 x timeout budget rule of `op_identities`. Expanding or factoring may not introduce functions other
    than the input's (and their companions: sin/cos, sinh/cosh). The chain that is reported is re-verified step
    by step and end to end with the symbolic zero test added, and the two kinds of evidence are reported
    separately. Statuses: `ok`, `unchanged` (nothing verified is better; says that this does not show that none
    exists), `conditions_not_satisfied` (the only improving identities are blocked), `unsupported`,
    `parse_error`, `resource_limit`.
  - `series` (`series.py`): Taylor polynomial about a finite real point, order 1 to 12, from `sympy.series`.
    Shown only if both checks pass, computed from the original function with mpmath at 80 digits at random
    values of every other symbol: (1) the coefficients against a Cauchy-integral estimate (64-node trapezoid
    rule on circles; trusted at the first radius where the estimates at r and r/2 agree, so a pole inside a
    circle is never mistaken for agreement, tolerance per coefficient so a wrong high-order term cannot hide on
    a small circle), and (2) the remainder |f - P| falling at least as h^(order+1) along real (and, for a
    variable that may be complex, imaginary) directions. A pole, branch point, logarithm or `|x|` is
    `unsupported`; a failed leg is `not_verified` and shows no formula. The measured-error table is a
    measurement at a few distances for expressions with no other symbols, not a bound.
  - `integrate` (`integrate.py`): `sympy.integrate` proposes an antiderivative; it is accepted only if a
    40-digit numerical derivative of it (real and, for a complex variable, imaginary direction) agrees with
    the integrand at assumption-respecting points. `Piecewise` results are checked branch by branch
    (equality cases are substituted into both sides; a case that cannot be sampled gives `not_verified`).
    A leftover `Integral` is `unsupported` ("no closed form found; this does not prove none exists").
    `simplify(F' - f) == 0` is reported separately as extra evidence. With both bounds (numbers, `oo`,
    `-oo`) it computes a definite integral instead and verifies it by independent adaptive quadrature
    (mpmath, 30 digits) at sampled parameter values, not by the fundamental theorem, which fails for a pole
    inside the interval. A quadrature with a large error estimate, or a non-finite or huge integrand on an
    interior grid, never counts as agreement; SymPy's convergence conditions that the declared assumptions
    do not imply give `conditions_not_satisfied`; divergent or undefined results show no value.
  - `matrix_calculus` (`matrix/`, `Engine.matrix_calculus`): evaluate, simplify, differentiate or integrate an
    expression in declared, noncommuting matrices (an abstract size or explicit entries) and their exponentials.
    `expressions.py` turns the parsed SymPy tree (declared names are noncommutative symbols) into a normal form
    (sum of scalar x ordered product of matrices and exponentials of sums of scalar multiples of matrices) and
    rejects everything else as `unsupported`; dimensions are checked on every product and sum. `methods.py` is a
    deterministic registry tried in a fixed order: `exp_explicit`, `exp_merge_commuting`, `exp_derivative`,
    `integrate_polynomial`, `exp_integral_invertible`, `sylvester_reduction` (the equation
    `aPX + cXQ = F(b) - F(a)` for `X = int e^{(at+b)P} C e^{(ct+d)Q}`, solved when the Kronecker operator is
    provably invertible), `block_exponential` (Van Loan: `X` from one block-matrix exponential, no invertibility
    or diagonalisability) and, last, the equation alone. Each method states its pattern, requirements and limits
    and verifies its own result with `verify.py` (numeric leg decides; exact series leg reported separately).
    The language model only fills in the declarations and explains the card; handbook `method.matrix.*` entries
    (stated) document the techniques and are retrieved by the features the engine emits (`matrix:*`).
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
    the equations of motion.
  - `invariants.py` (`search_invariants`): constants of motion beyond the Noether families. The ansatz
    C = Σ f_α(q) q̇^α (|α| ≤ D; f a product of powers q^a, |a| ≤ P, and for declared angles 1, cos, sin up to a
    harmonic) turns dC/dt = 0 on the equations of motion into a linear system for the coefficients over the field
    of the parameters (exact, no floats; cos²+sin²=1 is reduced). The nullspace modulo the known constants
    (cyclic momenta, energy, rotation charges, 1 and their products in the ansatz) is reported, and each
    candidate is checked again with the same `verify_constant` as `find_conserved` (symbolic zero and numeric
    points); anything that fails is dropped. It reports the exact family searched and that finding none proves
    nothing beyond it. Autonomous Lagrangians with rational accelerations only (square roots are `unsupported`),
    at most 3 coordinates and 500 unknowns (`resource_limit` above).
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
  `identities.simplify_expression` / `transform_expression` do the same for chains: candidates come from
  both sides of each rule (`SearchQuery.sides="both"`, an `rhs_features` FTS5 column that ordinary search
  ignores) plus a second round for the functions those rules can introduce; the engine runs the search and
  every cited rule is returned with its handbook entry (`entries`). If the search itself is hard-stopped,
  `simplify_expression` falls back to SymPy's simplification alone and says so.
  The matcher (`engine/ops.py`) discards matches that invent roots, reciprocals or `I`, and matches that
  bind a variable used only as `n*w` (like the `a` in `sin(2*a)`) to a fraction of the input.
- Generated entries (`generate.py`): `data/entries/generated_*.yaml` are written by
  `python -m minerva_knowledge generate` from seeds in `generate.py`; edit the seed, never the YAML.
  A candidate is written only if it round-trips through the text parser, validates, is not a duplicate,
  passes its own check (`symbolic` only with a symbolic zero) and the engine applies it to its own
  left-hand side. The test suite fails when the committed files differ from a fresh generation.
- Benchmark (`benchmark.py`, `data/benchmark/identity_cases.yaml`, `transform_cases.yaml`): hand-written
  expression → expected/forbidden identities, retrieval rank/recall/MRR over the same cases, and multi-step
  cases (goal, accepted forms, rules that must or must not appear, SymPy-simplify-only baseline), run with
  `python -m minerva_knowledge benchmark`; results and limits in `docs/handbook-benchmark.md`.

## C. The conversational agent (`services/api/src/minerva_api/agent/`)

The main interface is a chat. The language model plans, asks clarifying questions, searches the handbook
and explains; it never computes and never decides correctness.

- **Loop** (`loop.py`): the conversation and the tool schemas go to the model (`llm.py`: an Ollama client
  that streams `/api/chat`); requested tool calls run in the sandbox through `tools.execute`, results go back to the
  model, until it answers. A step cap and a time cap end runaway loops.
- **Tools** (`tools.py`): `search_handbook`, `get_entry`, `simplify`, `differentiate`, `integrate`, `series`, `matrix_calculus`, `find_identities`, `tensor_canonicalize`, `dimensional_analysis`,
  `check_units`, `solve_polynomial`, `analyze_lagrangian`, `search_invariants`, `expand_motion`, `perturbation_solve` and `integrate_motion`. Each is a
  pure computation. `integrate_motion` reuses the engine's request model as its schema, and the prompt tells the model never to
  choose parameter values, initial conditions or time spans itself. The Lagrangian card has a form for them that
  posts to `POST /v1/chat/conversations/{id}/tools/integrate_motion`: the model is not involved (it works with
  `MINERVA_LLM_MODE=off`), the user's numbers go to the engine untouched, and the result is stored as an assistant
  message with the card and a trace step, so later turns see it. A tool returns a `Card` that the interface renders from the tool's own output, plus a compact
  text for the model. Bad or unknown calls become an error result the model can read, never an exception.
- **Honesty** (`segments.py`): after each turn, every formula in the model's prose is compared with the formulas
  that tool results and handbook entries of the conversation contain. Others are shown as *unchecked*.
  Equations the model writes are then tested (`engine/claims.py`): an equation is marked *contradicted*
  only if its sides differ by a nonzero constant; one that is merely false at generic points stays *unchecked*
  (it may be an equation to solve). One with a symbol on one side only or a time derivative (a definition,
  coordinate change, constraint or equation of motion) is a relation, which sampling cannot judge, so it stays
  *unchecked* with a note, unless definitions stated earlier in the conversation decide it
  by substitution (the note names them).
- **Storage:** conversations and messages (per user, private); each assistant message stores its ordered blocks
  (prose and cards) and the tool trace so later turns have the same context.
- **Trace and debug report:** every agent request has a `trace_id` (header `X-Trace-ID`, every stream event, the stored
  message, the log). The loop records typed, content-free events (`agent/tracing.py`: `agent.started`, `llm.request`,
  `llm.response`, `tool.started`, `tool.completed`, `verification.completed`, `agent.retry`, `agent.stopped`,
  `agent.completed`) with monotonic timings, token counts when the host reports them, and a stop reason. Each stored
  assistant message carries `trace` (tool name, the arguments the model sent and the ones after `@rN` references were
  replaced, the text it read back, status, duration, an `evidence` record, an error category) and `debug` (trace id,
  model, prompt fingerprint, per-call timings, the events, how each equation the model wrote was tested). Failed turns
  are stored too. With "Debug mode" on, the chat shows a timeline and copies a Markdown report ending in a JSON block;
  `scripts/replay_trace.py` re-runs the recorded tool calls, or a whole turn with `--llm`, and the `chat-trace-triage`
  skill says how to classify the result. This is conversation content: it is shown and stored only with the
  conversation, under its access and deletion rules. The server log holds no content at any setting: `chat_turn` and
  the events carry identifiers, counts, durations and fixed vocabulary. See [observability.md](observability.md).
- **Reliability of the loop:** a turn ends with an explicit stop reason; an identical call is not run twice; a model call
  that fails before any output is retried once; unusable tool calls get one correction; verified results can be reused
  as `@rN` instead of being retyped; a formula is marked verified only if a verified result produced it (the user's own
  input, handbook entries that are only stated and identities that were not applied are marked *quoted*).
- **Models:** `MINERVA_LLM_MODE` is `off`, `ollama` (your own host: `MINERVA_LLM_BASE_URL`, `MINERVA_LLM_MODEL`) or
  `scripted` (deterministic stand-in for tests and UI work, development only). `scripts/eval_models.py` measures a
  model's tool-calling and honesty through the real loop on a fixed benchmark (`agent/evaluation/`: cases as data, scoring,
  failure classes, JSON and Markdown reports, baseline comparison; see [observability.md](observability.md)); interpretation
  in `docs/model-evaluation.md`.
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
limits. Endpoints: `/v1/auth/*`, `/v1/compute/{simplify (with `goal`, `target`),differentiate,integrate,series,matrix,identities,tensor/canonicalize}`,
`/v1/compute/apply-identity`, `/v1/derivations`, `/v1/reference/*` (public, read-only), `/v1/history`.
History rows contain user mathematics and are private to the account. A Workbench derivation is a
`derivations` row plus the calculations saved with its id; each step may name the step it continued
(`parent_id`). The server never accepts a result from the client: steps are written by the compute
endpoints, and `apply-identity` recomputes and re-verifies the rewrite from the handbook rule. The
client continues a step from its `result_text` (text syntax), because engine LaTeX does not always
read back as the same expression. Logging uses an allow-list of fields (no inputs, emails or tokens).

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
