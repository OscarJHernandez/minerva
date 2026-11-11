"""Application factory."""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from minerva_engine import Engine, Limits
from minerva_knowledge import KnowledgeBase
from sqlalchemy import text

from . import routers_auth, routers_chat, routers_compute, routers_history, routers_reference
from .agent.llm import LLMClient, OllamaClient
from .agent.scripted import ScriptedLLM
from .db import Base, make_engine, make_session_factory
from .logging import configure_logging, get_logger
from .security import RateLimiter
from .settings import Settings, get_settings

UNSAFE = {"POST", "PUT", "PATCH", "DELETE"}


def _make_llm(settings: Settings) -> LLMClient | None:
    if settings.llm_mode == "ollama":
        return OllamaClient(settings.llm_base_url, settings.llm_model, settings.llm_timeout_seconds)
    if settings.llm_mode == "scripted":
        return ScriptedLLM()
    return None


def create_app(settings: Settings | None = None, *, create_tables: bool = False) -> FastAPI:
    settings = settings or get_settings()
    settings.validate_for_runtime()
    configure_logging()
    log = get_logger()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        db_engine = make_engine(settings.database_url)
        if create_tables:  # tests and quick local use; production uses Alembic migrations
            Base.metadata.create_all(db_engine)
        app.state.db_engine = db_engine
        app.state.session_factory = make_session_factory(db_engine)
        app.state.kb = KnowledgeBase.load()
        app.state.engine = Engine(
            Limits(
                max_input_chars=settings.max_input_chars,
                timeout_seconds=settings.compute_timeout_seconds,
                memory_mb=settings.compute_memory_mb,
                max_concurrency=settings.compute_max_concurrency,
            )
        )
        app.state.llm = _make_llm(settings)
        await run_in_threadpool(app.state.engine.warm_up)
        log.info("startup", status="ready")
        yield
        db_engine.dispose()

    app = FastAPI(
        title="Minerva API",
        version="0.1.0",
        description="Mathematical computing API: verified simplification, identities, handbook.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.limiter = RateLimiter()

    @app.middleware("http")
    async def request_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = uuid.uuid4().hex
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        origin = request.headers.get("origin")
        if request.method in UNSAFE and origin and origin.rstrip("/") not in settings.origins:
            return JSONResponse({"detail": "Cross-origin request rejected"}, status_code=403)
        start = time.monotonic()
        response = await call_next(request)
        route = getattr(request.scope.get("route"), "path", "unmatched")
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path.startswith("/v1/") and not route.startswith("/v1/reference"):
            response.headers["Cache-Control"] = "no-store"
        log.info(
            "request",
            method=request.method,
            route=route,
            status_code=response.status_code,
            duration_ms=int((time.monotonic() - start) * 1000),
        )
        return response

    @app.get("/healthz", tags=["health"])
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", tags=["health"])
    def readyz(request: Request) -> JSONResponse:
        try:
            with request.app.state.db_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception:
            return JSONResponse({"status": "unavailable"}, status_code=503)
        return JSONResponse({"status": "ready"})

    for module in (
        routers_auth,
        routers_compute,
        routers_history,
        routers_reference,
        routers_chat,
    ):
        app.include_router(module.router)
    return app


def app_from_env() -> FastAPI:
    """Entry point for uvicorn: `uvicorn minerva_api.main:app_from_env --factory`."""
    return create_app()
