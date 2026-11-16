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

from . import (
    routers_auth,
    routers_chat,
    routers_compute,
    routers_derivations,
    routers_history,
    routers_reference,
)
from .agent.llm import LLMClient, OllamaClient
from .agent.scripted import ScriptedLLM
from .agent.vision import FormulaReader, OllamaFormulaReader, ScriptedFormulaReader
from .db import Base, make_engine, make_session_factory
from .logging import configure_logging, error_fields, get_logger
from .security import RateLimiter
from .settings import Settings, get_settings
from .softdelete import maybe_purge

UNSAFE = {"POST", "PUT", "PATCH", "DELETE"}


def _make_llm(settings: Settings) -> LLMClient | None:
    """The local model (Ollama or the dev stand-in). OpenRouter clients are made per request."""
    if settings.ollama_configured:
        return OllamaClient(
            settings.llm_base_url,
            settings.llm_model,
            settings.llm_timeout_seconds,
            think=settings.llm_think,
            strip_thinking=settings.llm_strip_thinking,
        )
    if settings.llm_mode == "scripted":
        return ScriptedLLM()
    return None


def _make_vision(settings: Settings) -> FormulaReader | None:
    if settings.ollama_configured and settings.llm_vision_model:
        return OllamaFormulaReader(
            settings.llm_base_url,
            settings.llm_vision_model,
            settings.llm_timeout_seconds,
            num_ctx=settings.llm_num_ctx,
        )
    if settings.llm_mode == "scripted":
        return ScriptedFormulaReader()
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
        with app.state.session_factory() as housekeeping:
            maybe_purge(housekeeping)  # deletions past their undo window become final
        app.state.kb = KnowledgeBase.load()
        app.state.engine = Engine(
            Limits(
                max_input_chars=settings.max_input_chars,
                timeout_seconds=settings.compute_timeout_seconds,
                memory_mb=settings.compute_memory_mb,
                max_concurrency=settings.compute_max_concurrency,
            )
        )
        app.state.local_llm = _make_llm(settings)
        app.state.llm = app.state.local_llm  # kept for older callers
        app.state.vision = _make_vision(settings)
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
        request.state.request_id = request_id
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        origin = request.headers.get("origin")
        if request.method in UNSAFE and origin and origin.rstrip("/") not in settings.origins:
            return JSONResponse({"detail": "Cross-origin request rejected"}, status_code=403)
        start = time.monotonic()
        try:
            response = await call_next(request)
        except Exception:
            # The exception goes on to the handler below; the request still gets its line.
            log.info(
                "request",
                method=request.method,
                route=getattr(request.scope.get("route"), "path", "unmatched"),
                status_code=500,
                duration_ms=int((time.monotonic() - start) * 1000),
            )
            raise
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

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
        # Fixed body and a log line with the exception's class and where our code raised it. Its message
        # can hold the user's expression or input, so it is never written or returned.
        log.error(
            "unhandled_error",
            route=getattr(request.scope.get("route"), "path", "unmatched"),
            **error_fields(exc),
        )
        return JSONResponse(
            {
                "detail": "Internal server error",
                "request_id": getattr(request.state, "request_id", None),
            },
            status_code=500,
        )

    @app.get("/healthz", tags=["health"])
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/version", tags=["health"])
    def version() -> dict[str, str]:
        # Public on purpose: a commit id is not sensitive, and the deploy test and a person can then check
        # which build is actually serving (`/api/v1/version` through the web app).
        return {"build": settings.build_sha}

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
        routers_derivations,
        routers_reference,
        routers_chat,
    ):
        app.include_router(module.router)
    return app


def app_from_env() -> FastAPI:
    """Entry point for uvicorn: `uvicorn minerva_api.main:app_from_env --factory`."""
    return create_app()
