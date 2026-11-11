# Minerva

An open-source mathematical computing assistant: **verified symbolic computation** plus a
**citable handbook** of mathematical identities, tables and physical equations.

Minerva is a calculator and reference for scientists, engineers and mathematicians, not a tutoring
app. It simplifies expressions, finds the identities that apply to them, and states exactly what
each one assumes. If something cannot be simplified or verified, it says so instead of guessing.

> **Status: early (milestone 1).** A small, working vertical slice. See [docs/roadmap.md](docs/roadmap.md)
> for what exists and what is planned.

## What works today

| Capability | How it is done |
| --- | --- |
| Simplify an expression | SymPy, then an independent numeric check at random sample points plus a symbolic zero test. Unverified results are never shown |
| Find applicable identities | SQLite FTS5 retrieval proposes candidates; the engine matches patterns, checks validity conditions against your assumptions, rewrites and verifies |
| Handbook | ~100 entries (algebra, calculus, series, special functions, tensors, mechanics, electromagnetism, thermodynamics, quantum mechanics, relativity, constants), each verified at a declared level or marked "stated" |
| Tensor canonicalization | `sympy.tensor.tensor` (monoterm symmetries only), HTTP API |
| Accounts and history | Email + password (Argon2id), server-side sessions, private per-user history |
| LaTeX input | MathLive editor or raw LaTeX, strict parser (no `eval`) |

Not implemented yet: Cadabra / SageManifolds backends, derivatives/integrals/limits in the UI,
semantic search, any LLM. All are on the roadmap and clearly labelled as planned.

## Quick start (Docker)

Requirements: Docker with Compose v2.

```bash
cp .env.example .env
# edit .env: set POSTGRES_PASSWORD and MINERVA_SESSION_SECRET to random values, e.g.
#   python3 -c "import secrets; print(secrets.token_urlsafe(48))"
docker compose up --build
```

- Landing page: <http://localhost:3001>
- Workspace: <http://localhost:3000> (create an account on the register page)
- API docs: <http://localhost:8100/docs>

## Development without Docker

Requirements: [uv](https://docs.astral.sh/uv/), Node 20.19+ (24 LTS recommended), pnpm 12, and
optionally PostgreSQL 17 (tests and local runs default to SQLite).

```bash
uv sync && pnpm install

# API (SQLite file for local use)
export MINERVA_DATABASE_URL=sqlite+pysqlite:///./dev.sqlite
(cd services/api && uv run alembic upgrade head)
uv run --directory services/api uvicorn minerva_api.main:app_from_env --factory --port 8100

# web app (second terminal)
MINERVA_API_URL=http://localhost:8100 pnpm --filter @minerva/web dev
```

Tests and checks:

```bash
uv run pytest                                   # Python (engine, knowledge, API)
uv run python -m minerva_knowledge verify       # verify every handbook entry
uv run ruff check . && uv run mypy packages/engine/src packages/knowledge/src services/api/src
pnpm lint && pnpm typecheck && pnpm test        # JavaScript
```

Postgres integration: `MINERVA_TEST_DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/db uv run pytest services`.

## Repository layout

```
apps/site/            landing page (static export)       packages/engine/     symbolic engine
apps/web/             app: workspace, handbook, login     packages/knowledge/  handbook + retrieval
packages/ui/          shared theme and math rendering     services/api/        FastAPI service
deploy/               Dockerfiles, Helm chart             docs/                design and policy docs
```

Details: [docs/architecture.md](docs/architecture.md). Contributor guide: [CONTRIBUTING.md](CONTRIBUTING.md).

## Adding to the handbook

Entries are YAML under `packages/knowledge/src/minerva_knowledge/data/entries/`. A new identity is a
few lines plus the test suite verifying it at the level you declare. See CONTRIBUTING.md.

## Licence

Apache-2.0 (see [LICENSE](LICENSE)), **provisional**: the assessment around GPL-licensed integrations
(Cadabra 2, SageMath) is in [docs/licensing.md](docs/licensing.md).

## Security

See [SECURITY.md](SECURITY.md). Computation runs in isolated, resource-limited processes and user
input is never executed as code.
