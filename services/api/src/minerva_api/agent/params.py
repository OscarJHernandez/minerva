"""Model parameters: operator defaults, per-turn overrides, and what a model host is sent.

One typed object serves three places: the server settings (`MINERVA_LLM_*`), the chat's debug panel
(an override for the next turn) and the evaluation script (flags that are recorded in the report).
Every field has a range the server enforces. Nothing mathematical is here: verification tolerances,
the tool-call cap, the step and time limits and the claim check are not parameters, and a test pins
the field list. These values change how the model writes and what it is shown, never what Minerva
accepts as a result.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..settings import Settings

Toolset = Literal["auto", "core", "all"]


class ParamOverrides(BaseModel):
    """What a request may change for one turn. A missing field keeps the server default."""

    model_config = ConfigDict(extra="forbid")

    num_ctx: int | None = Field(default=None, ge=2048, le=262144)
    num_predict: int | None = Field(default=None, ge=0, le=65536)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    top_k: int | None = Field(default=None, ge=1, le=1000)
    think: bool | None = None
    history_messages: int | None = Field(default=None, ge=2, le=200)
    history_tokens: int | None = Field(default=None, ge=256, le=262144)
    toolset: Toolset | None = None


class EffectiveParams(BaseModel):
    """The values a turn actually used. Numbers and fixed words only: stored with the reply."""

    num_ctx: int
    num_predict: int | None  # None: the model's own limit
    temperature: float
    top_p: float | None
    top_k: int | None
    think: bool
    history_messages: int
    history_tokens: int | None  # None: only the window limits the history
    toolset: Toolset

    def ollama_options(self) -> dict[str, Any]:
        """The `options` of an Ollama chat request; a value that is None is not sent."""
        options: dict[str, Any] = {"num_ctx": self.num_ctx, "temperature": self.temperature}
        for key in ("num_predict", "top_p", "top_k"):
            value = getattr(self, key)
            if value is not None:
                options[key] = value
        return options


# Only these can ever be parameters. A test compares the models' fields with this list so that a
# verification tolerance or a limit cannot be added here by accident.
PARAM_FIELDS = (
    "num_ctx",
    "num_predict",
    "temperature",
    "top_p",
    "top_k",
    "think",
    "history_messages",
    "history_tokens",
    "toolset",
)


def defaults(settings: Settings) -> EffectiveParams:
    return EffectiveParams(
        num_ctx=settings.llm_num_ctx,
        num_predict=settings.llm_num_predict or None,
        temperature=settings.llm_temperature,
        top_p=settings.llm_top_p or None,
        top_k=settings.llm_top_k or None,
        think=settings.llm_think,
        history_messages=settings.llm_history_messages,
        history_tokens=settings.llm_history_tokens or None,
        toolset=settings.llm_toolset,
    )


class ParamError(ValueError):
    """An override the account may not make. The message is safe to show."""


def may_tune(settings: Settings, email: str) -> bool:
    """Whether this account may override parameters other than the context size."""
    return settings.llm_tuning or may_set_context(settings, email)


def may_set_context(settings: Settings, email: str) -> bool:
    allowed = settings.llm_tuning_user_set
    return "*" in allowed or email.strip().lower() in allowed


def resolve(settings: Settings, overrides: ParamOverrides | None, email: str) -> EffectiveParams:
    """Defaults with the account's overrides applied; refuses what the account may not change."""
    base = defaults(settings)
    if overrides is None:
        return base
    given = overrides.model_dump(exclude_none=True)
    if not given:
        return base
    if not may_tune(settings, email):
        raise ParamError("Changing model parameters is not enabled for this account.")
    if "num_ctx" in given:
        if not may_set_context(settings, email):
            raise ParamError("Changing the context size is not allowed for this account.")
        if given["num_ctx"] > settings.llm_num_ctx_max:
            raise ParamError(f"The context size is limited to {settings.llm_num_ctx_max} tokens.")
    update = dict(given)
    if update.get("num_predict") == 0:
        update["num_predict"] = None
    return base.model_copy(update=update)


def describe(settings: Settings, email: str) -> dict[str, Any]:
    """What `/v1/chat/status` tells the chat: defaults, which fields this account may change."""
    tune = may_tune(settings, email)
    return {
        "defaults": defaults(settings).model_dump(),
        "editable": [
            f for f in PARAM_FIELDS if tune and (f != "num_ctx" or may_set_context(settings, email))
        ],
        "num_ctx_max": settings.llm_num_ctx_max,
    }
