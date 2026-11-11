"""Registration, login, logout."""

from __future__ import annotations

from datetime import timedelta

from fastapi import APIRouter, HTTPException, Request, Response, status
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError

from .db import AuthSession, User, utcnow
from .deps import DbDep, LimiterDep, SettingsDep, UserDep, client_key
from .logging import get_logger
from .schemas import Credentials, UserOut
from .security import (
    hash_password,
    needs_rehash,
    new_session_token,
    token_digest,
    verify_password,
)

router = APIRouter(prefix="/v1/auth", tags=["auth"])
log = get_logger()


def _start_session(response: Response, db: DbDep, settings: SettingsDep, user: User) -> None:
    token = new_session_token()
    expires = utcnow() + timedelta(hours=settings.session_ttl_hours)
    db.add(
        AuthSession(
            user_id=user.id,
            token_hash=token_digest(token, settings.session_secret.get_secret_value()),
            expires_at=expires,
        )
    )
    db.commit()
    response.set_cookie(
        settings.cookie_name,
        token,
        max_age=settings.session_ttl_hours * 3600,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
        path="/",
    )


def _throttle(request: Request, limiter: LimiterDep, settings: SettingsDep, scope: str) -> None:
    if not limiter.allow(f"{scope}:{client_key(request)}", settings.auth_rate_per_minute):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Too many attempts; slow down.")


@router.post("/register", status_code=status.HTTP_201_CREATED, response_model=UserOut)
def register(
    body: Credentials,
    request: Request,
    response: Response,
    db: DbDep,
    settings: SettingsDep,
    limiter: LimiterDep,
) -> UserOut:
    _throttle(request, limiter, settings, "register")
    if not settings.allow_registration:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Registration is disabled.")
    email = body.email.lower()
    user = User(email=email, password_hash=hash_password(body.password))
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "Could not create that account.") from None
    _start_session(response, db, settings, user)
    log.info("user_registered", user_id=str(user.id))
    return UserOut(id=user.id, email=user.email)


@router.post("/login", response_model=UserOut)
def login(
    body: Credentials,
    request: Request,
    response: Response,
    db: DbDep,
    settings: SettingsDep,
    limiter: LimiterDep,
) -> UserOut:
    _throttle(request, limiter, settings, "login")
    user = db.scalar(select(User).where(User.email == body.email.lower()))
    ok = verify_password(user.password_hash if user else None, body.password)
    if not ok or user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid email or password.")
    if needs_rehash(user.password_hash):
        user.password_hash = hash_password(body.password)
        db.commit()
    db.execute(delete(AuthSession).where(AuthSession.expires_at < utcnow()))
    _start_session(response, db, settings, user)
    return UserOut(id=user.id, email=user.email)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(request: Request, response: Response, db: DbDep, settings: SettingsDep) -> None:
    token = request.cookies.get(settings.cookie_name)
    if token:
        digest = token_digest(token, settings.session_secret.get_secret_value())
        db.execute(delete(AuthSession).where(AuthSession.token_hash == digest))
        db.commit()
    response.delete_cookie(settings.cookie_name, path="/")


@router.get("/me", response_model=UserOut)
def me(user: UserDep) -> UserOut:
    return UserOut(id=user.id, email=user.email)
