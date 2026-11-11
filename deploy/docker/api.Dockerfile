# syntax=docker/dockerfile:1
# Build context: repository root.
FROM python:3.15-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
COPY packages packages
COPY services services
# Non-editable install of the API and its workspace dependencies, without dev tools.
RUN uv sync --frozen --no-dev --no-editable --package minerva-api

FROM python:3.15-slim AS runtime
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN groupadd --system --gid 10001 minerva \
 && useradd --system --uid 10001 --gid minerva --no-create-home --shell /usr/sbin/nologin minerva
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY services/api/alembic.ini services/api/alembic.ini
COPY services/api/alembic services/api/alembic
USER 10001:10001
EXPOSE 8100
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8100/healthz', timeout=3).status == 200 else 1)"
# Migrations run first; the app refuses to start outside development with placeholder secrets.
CMD ["sh", "-c", "alembic -c services/api/alembic.ini upgrade head && exec uvicorn minerva_api.main:app_from_env --factory --host 0.0.0.0 --port 8100 --no-server-header"]
