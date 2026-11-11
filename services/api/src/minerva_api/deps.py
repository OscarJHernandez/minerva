"""FastAPI dependencies: settings, database session, authenticated user."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from minerva_engine import Engine
from minerva_knowledge import KnowledgeBase
from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import AuthSession, User
from .security import RateLimiter, token_digest
from .settings import Settings


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings  # type: ignore[no-any-return]


def get_db(request: Request) -> Iterator[Session]:
    session: Session = request.app.state.session_factory()
    try:
        yield session
    finally:
        session.close()


def get_engine(request: Request) -> Engine:
    return request.app.state.engine  # type: ignore[no-any-return]


def get_kb(request: Request) -> KnowledgeBase:
    return request.app.state.kb  # type: ignore[no-any-return]


def get_limiter(request: Request) -> RateLimiter:
    return request.app.state.limiter  # type: ignore[no-any-return]


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
DbDep = Annotated[Session, Depends(get_db)]
EngineDep = Annotated[Engine, Depends(get_engine)]
KbDep = Annotated[KnowledgeBase, Depends(get_kb)]
LimiterDep = Annotated[RateLimiter, Depends(get_limiter)]


def current_user(request: Request, db: DbDep, settings: SettingsDep) -> User:
    token = request.cookies.get(settings.cookie_name)
    unauthorized = HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated")
    if not token:
        raise unauthorized
    digest = token_digest(token, settings.session_secret.get_secret_value())
    row = db.scalar(select(AuthSession).where(AuthSession.token_hash == digest))
    if row is None:
        raise unauthorized
    expires = row.expires_at if row.expires_at.tzinfo else row.expires_at.replace(tzinfo=UTC)
    if expires <= datetime.now(UTC):
        db.delete(row)
        db.commit()
        raise unauthorized
    return row.user


UserDep = Annotated[User, Depends(current_user)]


def client_key(request: Request) -> str:
    # Behind Cloudflare/ingress the real client is in a forwarded header; deployments must
    # only trust it from their own proxy. Locally this falls back to the socket peer.
    forwarded = request.headers.get("cf-connecting-ip") or request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()[:64]
    return request.client.host if request.client else "unknown"
