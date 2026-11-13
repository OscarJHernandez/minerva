---
name: verify-change
description: Run the right checks before declaring Minerva work done. Use at the end of any change, before a commit or pull request, or when asked whether something is ready. Maps changed paths to the checks CI will run.
---

# verify-change

Report what you ran and what it printed. "Should pass" is not a result. If a check cannot run here
(no PostgreSQL, no browser), say so rather than skipping silently.

## 1. See what changed

```bash
git status --short && git diff --stat
```

## 2. Run the checks for those paths

| Changed paths | Run |
| --- | --- |
| `packages/engine`, `packages/knowledge`, `services/api`, `pyproject.toml`, `uv.lock` | `make check-py` |
| `packages/knowledge/.../data` (handbook) | `make handbook`, plus the `math-regression` skill |
| `apps/`, `packages/ui`, `package.json`, `pnpm-lock.yaml` | `make check-web`, plus the `ui-check` skill |
| `services/api/alembic`, database models | `db-migration` skill (includes `alembic check`) |
| `.github/workflows` | `ci-workflows` skill |
| `deploy/`, `scripts/` | `bash -n scripts/build-image.sh`; Helm lint per `.github/workflows/ci.yml` (`lint` job) |
| anything | `make check` runs all of the above except Helm and images |

`make check` is the same set as the CI jobs `python`, `tests`, `tests-api` and `web`. CI splits the pytest run over
runner pods with `MINERVA_TEST_SHARD=i/n` (engine, handbook and scripts in 5 shards, API tests in 2); unset runs everything.

## 3. Definition of done (`AGENTS.md`)

1. Behaviour is real computation; unsupported cases return an explicit status.
2. Tests: unit and regression (expected values, negative cases, assumption cases).
3. Lint, types and tests pass; `python -m minerva_knowledge verify` passes.
4. No secrets, user data or copyrighted text; licence impact of new dependencies checked.
5. Docs updated (`AGENTS.md` if commands or architecture changed, `docs/` for design decisions).
6. UI changes checked in a browser in light and dark at phone width.

## 4. Report

List: checks run with pass/fail, checks not run and why, and anything the maintainer must decide.
