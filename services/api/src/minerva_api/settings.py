"""Runtime configuration, read from MINERVA_* environment variables."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MINERVA_", env_file=".env", extra="ignore")

    env: str = "development"
    database_url: str = "sqlite+pysqlite:///:memory:"
    # Placeholder default is rejected outside development (see `validate_for_runtime`).
    session_secret: SecretStr = SecretStr("change-me-generate-a-random-value")
    session_ttl_hours: int = 24 * 14
    cookie_secure: bool = False
    cookie_name: str = "minerva_session"
    allowed_origins: str = "http://localhost:3000"
    allow_registration: bool = True

    compute_timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    compute_memory_mb: int = Field(default=1024, ge=128, le=16384)
    compute_max_concurrency: int = Field(default=4, ge=1, le=64)
    max_input_chars: int = Field(default=2000, ge=10, le=20000)

    # Header carrying the real client IP, set by a proxy you control (e.g. cf-connecting-ip behind
    # Cloudflare). Empty means: use the socket peer. Never trust a header that clients can set.
    trusted_ip_header: str = ""
    compute_rate_per_minute: int = 60
    auth_rate_per_minute: int = 10

    # Chat agent. "off": chat disabled. "scripted": deterministic stand-in (tests, UI development).
    # "ollama": your own Ollama-compatible host given by llm_base_url / llm_model.
    llm_mode: Literal["off", "scripted", "ollama"] = "off"
    llm_base_url: str = ""
    llm_model: str = ""
    llm_timeout_seconds: float = Field(default=120.0, gt=0, le=900)
    llm_think: bool = False  # let reasoning models think before replying (slower)
    llm_strip_thinking: bool = (
        False  # drop reasoning a model leaks into its reply (delays streaming)
    )
    # Reading formulas from a picture ("Add image"): a vision-capable model on the same Ollama host
    # (it may be the chat model if that accepts images). Empty: the feature is off.
    llm_vision_model: str = ""
    image_max_bytes: int = Field(default=4_000_000, ge=10_000, le=20_000_000)
    image_rate_per_minute: int = 6
    agent_max_steps: int = Field(default=6, ge=1, le=12)
    agent_max_seconds: float = Field(default=180.0, gt=0, le=900)
    agent_max_tool_calls: int = Field(default=12, ge=1, le=40)
    # What the agent writes to the server log per request: "off" nothing beyond the request lines,
    # "summary" one `agent.completed` event, "all" every trace event. Events never hold content
    # (expressions, tool arguments or results, model text); see docs/observability.md.
    agent_events: Literal["off", "summary", "all"] = "summary"
    chat_rate_per_minute: int = 20

    @field_validator("allowed_origins")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()

    @property
    def origins(self) -> list[str]:
        return [o.strip().rstrip("/") for o in self.allowed_origins.split(",") if o.strip()]

    def validate_for_runtime(self) -> None:
        """Refuse to start in production with placeholder or weak secrets."""
        if self.llm_mode == "ollama" and not (self.llm_base_url and self.llm_model):
            raise RuntimeError(
                "MINERVA_LLM_MODE=ollama needs MINERVA_LLM_BASE_URL and MINERVA_LLM_MODEL"
            )
        if self.llm_mode == "scripted" and self.env.lower() != "development":
            raise RuntimeError("the scripted stand-in model is for development only")
        if self.env.lower() == "development":
            return
        secret = self.session_secret.get_secret_value()
        if secret.startswith("change-me") or len(secret) < 32:
            raise RuntimeError("MINERVA_SESSION_SECRET must be a random value of 32+ characters")
        if not self.cookie_secure:
            raise RuntimeError("MINERVA_COOKIE_SECURE must be true outside development")


@lru_cache
def get_settings() -> Settings:
    return Settings()
