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
apps/web/             Authenticated app: chat (home), workbench (/workbench), handbook, login. Next.js 16 + Tailwind 4
packages/ui/          Shared theme tokens and the KaTeX <Math> component
packages/engine/      Symbolic engine (Python). Parsers, sandbox, simplify, rules, tensors, dimensions
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
uv run python -m minerva_knowledge benchmark          # hand-written expression -> identity cases (docs/handbook-benchmark.md)
uv run python -m minerva_knowledge generate [--check] # regenerate data/entries/generated_*.yaml from seeds in generate.py
uv run python scripts/eval_models.py --base-url http://HOST:11434 --models a,b --out docs/model-evaluation.md
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
- Conversations are user data: private per account, never logged.

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
- Tensor support is exactly what `sympy.tensor.tensor` does (monoterm symmetries). Multi-term
  identities (Bianchi) are out of scope and must be reported as such.
- User input is never executed: no `eval`/`exec`/`sympify`/`parse_expr`. Parsers are
  `latexparse.py` and `textparse.py`, building SymPy trees through `build.Builder`.
- All computation runs in the sandbox (`sandbox.py`): child process, wall-clock timeout, rlimits,
  concurrency cap. Add new operations to `ops.OPERATIONS` and call them via `Engine`.
- LaTeX conventions: `e` and `i` are plain symbols; `\exponentialE` / `\imaginaryI` / `\pi` are
  constants; `\sin 2x` and `\sin^{-1}` are rejected as ambiguous.
- In text syntax `E`, `I`, `pi`, `oo` are reserved constants: never use `E` as a variable name in
  handbook entries (use `En`, `Ef`, ...).

Use the `math-regression` skill when adding or changing any operation or handbook entry.

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
