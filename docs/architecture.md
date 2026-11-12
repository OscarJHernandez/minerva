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
  - `dimensions.py`: dimensional consistency of physics equations (used by handbook checks).
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
- **Tools** (`tools.py`): `search_handbook`, `get_entry`, `simplify`, `differentiate`, `find_identities`, `tensor_canonicalize`, plus
  `solve_polynomial` and `analyze_lagrangian`. Each is a
  pure computation. A tool returns a `Card` that the interface renders from the tool's own output, plus a compact
  text for the model. Bad or unknown calls become an error result the model can read, never an exception.
- **Honesty** (`segments.py`): after each turn, every formula in the model's prose is compared with the formulas
  that tool results and handbook entries of the conversation contain. Others are shown as *unchecked*.
- **Storage:** conversations and messages (per user, private); each assistant message stores its ordered blocks
  (prose and cards) and the tool trace so later turns have the same context.
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
limits. Endpoints: `/v1/auth/*`, `/v1/compute/{simplify,differentiate,identities,tensor/canonicalize}`,
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
