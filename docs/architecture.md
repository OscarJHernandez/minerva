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

## C. Optional LLM

Not implemented. Design: an `LLMClient` interface configured by `MINERVA_LLM_BASE_URL` /
`MINERVA_LLM_MODEL` (Ollama-compatible endpoint; unset means the feature is off). The model may
suggest candidate entries by id and explain results; it never computes, never decides applicability and
never writes results. Every suggestion goes through the same engine checks as retrieval candidates.

## API (`services/api`)

FastAPI with OpenAPI docs at `/docs`. Auth: Argon2id passwords, random session tokens stored as HMAC
digests, `HttpOnly` `SameSite=Lax` cookie, same-origin check on unsafe methods, per-process rate
limits. Endpoints: `/v1/auth/*`, `/v1/compute/{simplify,identities,tensor/canonicalize}`,
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
