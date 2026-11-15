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

import re
import time
from typing import Any, Literal

import httpx
from fastapi import HTTPException, Request, status
from pydantic import BaseModel, Field, SecretStr

from .agent.llm import LLMClient, OllamaClient
from .agent.openrouter import MODEL_ID, OpenRouterClient
from .db import User
from .security import RateLimiter
from .settings import Settings

DAY_SECONDS = 86400.0
PROBE_TTL = 30.0

# A model tag as Ollama writes it (`qwen3:8b`, `library/name:tag`). It is only ever sent as the
# `model` field of a request to the operator's own host, never used as an address.
LOCAL_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,99}$")

# Model families that Ollama documents as supporting tool calling, which Minerva's agent needs.
# This is guidance about capability, not a measurement: `docs/model-evaluation.md` has the scores.
RECOMMENDED_LOCAL: tuple[tuple[str, str], ...] = (
    ("qwen3", "Tool calling; good general maths for its size"),
    ("qwen2.5", "Tool calling; small sizes run on modest hardware"),
    ("llama3.1", "Tool calling; widely available"),
    ("llama3.3", "Tool calling; large, needs a strong GPU"),
    ("mistral-nemo", "Tool calling; mid-sized"),
    ("mistral-small", "Tool calling; mid-sized"),
    ("gpt-oss", "Tool calling; open-weight reasoning model"),
)


def recommended_family(name: str) -> str | None:
    """The recommended family an installed model tag belongs to, if any."""
    base = name.split(":", 1)[0].rsplit("/", 1)[-1].lower()
    for family, _ in RECOMMENDED_LOCAL:
        if base == family:
            return family
    return None


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
        wanted = choice.model if choice else None
        if wanted and wanted != local.name:
            return _other_local_model(request, settings, wanted), local.provider
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


def _other_local_model(request: Request, settings: Settings, model: str) -> LLMClient:
    """A client for another model on the operator's own host, only if that host has it installed.

    The list comes from the cached probe (`ollama_models`); when it is known and the model is not
    in it the request is refused, so a typo does not become a slow failing turn.
    """
    if not settings.ollama_configured:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "Only the configured model is available here."
        )
    if not LOCAL_MODEL_ID.match(model):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Choose a valid model name.")
    cache: tuple[float, list[str]] | None = getattr(request.app.state, "ollama_models", None)
    if cache and cache[1] and model not in cache[1]:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "That model is not installed on the server."
        )
    return OllamaClient(
        settings.llm_base_url,
        model,
        settings.llm_timeout_seconds,
        think=settings.llm_think,
        strip_thinking=settings.llm_strip_thinking,
    )


async def ollama_models(request: Request, settings: Settings) -> list[str] | None:
    """Model tags installed on the operator's Ollama host, or None if it cannot be reached.

    One cached probe (30 s) answers both "reachable?" and "which models?". The address comes from
    settings only.
    """
    if not settings.ollama_configured:
        return None
    cache: tuple[float, list[str] | None] | None = getattr(request.app.state, "ollama_probe", None)
    now = time.monotonic()
    if cache and now - cache[0] < PROBE_TTL:
        return cache[1]
    names: list[str] | None = None
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(3.0),
            follow_redirects=False,
            transport=getattr(request.app.state, "ollama_probe_transport", None),
        ) as client:
            reply = await client.get(f"{settings.llm_base_url.rstrip('/')}/api/tags")
        if reply.status_code == 200:
            try:
                rows = reply.json().get("models", [])
                names = sorted(
                    {
                        str(row["name"])
                        for row in rows
                        if isinstance(row, dict)
                        and isinstance(row.get("name"), str)
                        and LOCAL_MODEL_ID.match(row["name"])
                    }
                )
            except ValueError:
                names = []
    except httpx.HTTPError:
        names = None
    request.app.state.ollama_probe = (now, names)
    request.app.state.ollama_models = (now, names or [])
    return names


async def ollama_reachable(request: Request, settings: Settings) -> bool:
    """Is the operator's own Ollama host answering? (Cached; see `ollama_models`.)"""
    return await ollama_models(request, settings) is not None


def local_model_choices(installed: list[str], current: str) -> dict[str, Any]:
    """What the settings panel shows for the local provider: installed models, recommended first.

    `installed` comes from the host itself; nothing here names a model that is not there except
    the family hints in `suggested`, which are plain guidance and are never selectable.
    """
    names = sorted(set(installed) | ({current} if current else set()))
    rows = [
        {
            "name": name,
            "recommended": recommended_family(name) is not None,
            "note": dict(RECOMMENDED_LOCAL).get(recommended_family(name) or "", ""),
        }
        for name in names
    ]
    rows.sort(key=lambda r: (not r["recommended"], str(r["name"])))
    have = {recommended_family(n) for n in names}
    return {
        "models": rows,
        "suggested": [
            {"family": family, "note": note}
            for family, note in RECOMMENDED_LOCAL
            if family not in have
        ],
    }


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
