"""What model answers: asked of the model host, stamped on the reply, never guessed.

`minerva` is an alias the operator can re-point, so a name alone does not say what the user is talking to. The
weights (base model, size, quantisation, digest) are what the host reports for the model; a host that does not
answer gives `known=False` and the interface says the identity was not reported. Nothing here names a model the
host did not report, and the host's address never leaves the server.

Two uses with different freshness: the status chip may use a short cache, while the stamp on a reply is taken from
the host at the start of the turn and again when the reply is complete, because a reply sent just after the alias
was re-pointed must not be stamped with the old weights. Ollama's chat response carries no digest, so two small
requests around the turn are the closest the host allows.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from pydantic import BaseModel, Field

PROBE_TIMEOUT_SECONDS = 3.0


class ModelInfo(BaseModel):
    """The model behind one reply (stored with it) or behind the status chip."""

    provider: str  # ollama | openrouter | scripted
    alias: str  # the configured name, which may be re-pointed
    hosted: bool = (
        False  # a provider's model: weights, quantisation and serving are theirs and not reported
    )
    known: bool = False  # did the host report the weights?
    base_model: str | None = None
    family: str | None = None
    parameter_size: str | None = None
    quantization: str | None = None
    digest: str | None = None  # first 12 characters
    can_think: bool | None = None
    loaded_context: int | None = None
    # How this reply was asked for: filled from the turn's parameters.
    thinking_requested: bool | None = None
    reply_budget: int | None = None
    # The host served different weights between the start and the end of the reply.
    changed_during_reply: bool = False
    digest_after: str | None = None
    notes: list[str] = Field(default_factory=list)


def hosted_info(provider: str, alias: str) -> ModelInfo:
    return ModelInfo(
        provider=provider,
        alias=alias,
        hosted=True,
        notes=["hosted by the provider: weights, quantisation and serving settings are theirs"],
    )


def scripted_info() -> ModelInfo:
    return ModelInfo(
        provider="scripted",
        alias="scripted",
        notes=["a deterministic stand-in for development, not a language model"],
    )


def _short(digest: Any) -> str | None:
    return str(digest)[:12] if isinstance(digest, str) and digest else None


def info_from(
    alias: str, show: dict[str, Any], tags_entry: dict[str, Any], ps_entry: dict[str, Any]
) -> ModelInfo:
    """Build the identity from the three host answers (`/api/show`, one `/api/tags` row, one `/api/ps` row)."""
    raw_details = show.get("details")
    details: dict[str, Any] = raw_details if isinstance(raw_details, dict) else {}
    capabilities = show.get("capabilities")
    context = ps_entry.get("context_length")
    return ModelInfo(
        provider="ollama",
        alias=alias,
        known=True,
        base_model=details.get("parent_model") or None,
        family=details.get("family"),
        parameter_size=details.get("parameter_size"),
        quantization=details.get("quantization_level"),
        digest=_short(tags_entry.get("digest")),
        can_think="thinking" in capabilities if isinstance(capabilities, list) else None,
        loaded_context=context if isinstance(context, int) and context > 0 else None,
    )


def _row(payload: Any, alias: str) -> dict[str, Any]:
    rows = payload.get("models", []) if isinstance(payload, dict) else []
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict) and row.get("name") in (alias, f"{alias}:latest"):
            return row
    return {}


async def probe(
    base_url: str,
    alias: str,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    timeout: float = PROBE_TIMEOUT_SECONDS,
) -> ModelInfo:
    """Ask the host what `alias` is, now. Never raises: a host that does not answer gives `known=False`."""
    root = base_url.rstrip("/")
    unknown = ModelInfo(provider="ollama", alias=alias)
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout), follow_redirects=False, transport=transport
        ) as client:
            show, tags, ps = await asyncio.gather(
                client.post(f"{root}/api/show", json={"model": alias}),
                client.get(f"{root}/api/tags"),
                client.get(f"{root}/api/ps"),
            )
        if show.status_code != 200 or tags.status_code != 200:
            return unknown
        ps_payload = ps.json() if ps.status_code == 200 else {}
        return info_from(alias, show.json(), _row(tags.json(), alias), _row(ps_payload, alias))
    except (httpx.HTTPError, ValueError):
        return unknown


async def probe_digest(
    base_url: str,
    alias: str,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    timeout: float = PROBE_TIMEOUT_SECONDS,
) -> str | None:
    """Only the digest of `alias`, now (the end-of-reply check). None when the host does not say."""
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout), follow_redirects=False, transport=transport
        ) as client:
            tags = await client.get(f"{base_url.rstrip('/')}/api/tags")
        return _short(_row(tags.json(), alias).get("digest")) if tags.status_code == 200 else None
    except (httpx.HTTPError, ValueError):
        return None


def stamped(
    before: ModelInfo,
    after_digest: str | None,
    *,
    thinking_requested: bool | None,
    reply_budget: int | None,
) -> ModelInfo:
    """`before` with how the reply was asked for, and whether the weights changed while it was written."""
    changed = bool(before.digest and after_digest and before.digest != after_digest)
    return before.model_copy(
        update={
            "thinking_requested": thinking_requested,
            "reply_budget": reply_budget,
            "changed_during_reply": changed,
            "digest_after": after_digest if changed else None,
        }
    )
