#!/usr/bin/env bash
# The API for the browser tests: a throwaway SQLite database, the deterministic scripted model (development only,
# no real model and no network) and the same 10 s compute limit as the deployed chart.
set -euo pipefail
cd "$(dirname "$0")/.."
DIR="$(mktemp -d)"
trap 'rm -rf "$DIR"' EXIT
export MINERVA_DATABASE_URL="sqlite+pysqlite:///$DIR/e2e.sqlite"
export MINERVA_ENV=development
export MINERVA_COOKIE_SECURE=false
export MINERVA_SESSION_SECRET="e2e-only-secret-not-used-anywhere-else-0123456789"
export MINERVA_ALLOWED_ORIGINS="http://localhost:${E2E_WEB_PORT:-3131}"
export MINERVA_LLM_MODE=scripted
export MINERVA_COMPUTE_TIMEOUT_SECONDS=10
(cd services/api && uv run --no-sync alembic upgrade head >/dev/null)
uv run --no-sync --directory services/api uvicorn minerva_api.main:app_from_env --factory \
  --host 127.0.0.1 --port "${E2E_API_PORT:-8131}" --no-access-log
