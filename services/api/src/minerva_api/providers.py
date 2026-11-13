"""Which model answers a chat turn: Ollama (the server's own host) or OpenRouter (cloud).

Rules that are enforced here and nowhere else:
- Nothing falls back from one provider to another. A failing provider is reported as such.
- A cloud turn needs the user's explicit consent in the request (`cloud_consent`), unless the operator
  made OpenRouter the deployment default.
- A key typed by the user is accepted only when the operator enabled BYOK. The operator's own key is
  usable only by listed accounts and under a daily per-user cap.
- The OpenRouter address is a constant; users never choose a URL.
"""

from __future__ import annotations

import time
from typing import Any, Literal

import httpx
from fastapi import HTTPException, Request, status
from pydantic import BaseModel, Field, SecretStr

from .agent.llm import LLMClient
from .agent.openrouter import MODEL_ID, OpenRouterClient
from .db import User
from .security import RateLimiter
from .settings import Settings

DAY_SECONDS = 86400.0
PROBE_TTL = 30.0


class ProviderChoice(BaseModel):
    kind: Literal["ollama", "openrouter"]
    model: str | None = Field(default=None, max_length=100)
    api_key: SecretStr | None = Field(default=None, max_length=512, repr=False)
    cloud_consent: bool = False

    def __repr__(self) -> str:  # the key never appears, even in a debug print
        return f"ProviderChoice(kind={self.kind!r}, model={self.model!r})"


class OpenRouterTest(BaseModel):
    api_key: SecretStr | None = Field(default=None, max_length=512, repr=False)
    model: str = Field(min_length=1, max_length=100)


def _forbidden(detail: str) -> HTTPException:
    return HTTPException(status.HTTP_403_FORBIDDEN, detail)


def server_key_allowed(settings: Settings, user: User) -> bool:
    """May this account spend the operator's OpenRouter key?"""
    if not settings.has_server_key:
        return False
    allowed = settings.openrouter_key_user_set
    return "*" in allowed or user.email.lower() in allowed


def resolve_llm(
    request: Request,
    settings: Settings,
    user: User,
    limiter: RateLimiter,
    choice: ProviderChoice | None,
) -> tuple[LLMClient, str]:
    """The client for this turn, and the provider label. Raises HTTPException with safe text."""
    mode = settings.llm_mode
    if mode == "off":
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "The chat assistant is not configured."
        )
    explicit = choice is not None
    kind = choice.kind if choice else ("openrouter" if mode == "openrouter" else "local")
    if kind in ("ollama", "local"):
        local: LLMClient | None = request.app.state.local_llm
        if local is None:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE, "The local model is not configured."
            )
        return local, local.provider

    # OpenRouter
    if not settings.openrouter_available:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "OpenRouter is not enabled on this server."
        )
    if explicit and not (choice and choice.cloud_consent):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "Confirm that this conversation may be sent to OpenRouter.",
        )
    model = (choice.model if choice else None) or settings.openrouter_model
    if not model or not MODEL_ID.match(model):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Choose a valid model id.")
    user_key = choice.api_key.get_secret_value() if choice and choice.api_key else ""
    if user_key:
        if not settings.openrouter_byok:
            raise _forbidden("Personal OpenRouter keys are not enabled on this server.")
        key = user_key
    else:
        if not server_key_allowed(settings, user):
            raise _forbidden(
                "Add your own OpenRouter API key in Model settings to use this provider."
            )
        if not limiter.allow(
            f"orkey:{user.id}", settings.openrouter_daily_turns, window_seconds=DAY_SECONDS
        ):
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                "Daily limit for the shared OpenRouter key reached.",
            )
        key = settings.openrouter_api_key.get_secret_value()
    client = OpenRouterClient(
        key,
        model,
        settings.openrouter_timeout_seconds,
        max_tokens=settings.openrouter_max_tokens,
        data_collection=settings.openrouter_data_collection,
        zdr=settings.openrouter_zdr,
        transport=getattr(request.app.state, "openrouter_transport", None),
    )
    return client, "openrouter"


async def ollama_reachable(request: Request, settings: Settings) -> bool:
    """Ping the operator's own Ollama host (cached 30 s). The address comes from settings only."""
    if not settings.ollama_configured:
        return False
    cache: tuple[float, bool] | None = getattr(request.app.state, "ollama_probe", None)
    now = time.monotonic()
    if cache and now - cache[0] < PROBE_TTL:
        return cache[1]
    ok = False
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(3.0),
            follow_redirects=False,
            transport=getattr(request.app.state, "ollama_probe_transport", None),
        ) as client:
            ok = (
                await client.get(f"{settings.llm_base_url.rstrip('/')}/api/tags")
            ).status_code == 200
    except httpx.HTTPError:
        ok = False
    request.app.state.ollama_probe = (now, ok)
    return ok


def provider_status(settings: Settings, user: User | None) -> dict[str, Any]:
    return {
        "default": "openrouter" if settings.llm_mode == "openrouter" else "ollama",
        "ollama": {
            "available": settings.ollama_configured or settings.llm_mode == "scripted",
            "model": settings.llm_model,
        },
        "openrouter": {
            "available": settings.openrouter_available,
            "byok": settings.openrouter_byok,
            "server_key_for_you": user is not None and server_key_allowed(settings, user),
            "default_model": settings.openrouter_model,
            "models": settings.openrouter_model_list,
            "zdr": settings.openrouter_zdr,
            "data_collection": settings.openrouter_data_collection,
        },
    }
