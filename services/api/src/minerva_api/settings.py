"""Runtime configuration, read from MINERVA_* environment variables."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# A short starting list of tool-capable models; operators override it with MINERVA_OPENROUTER_MODELS and
# users can type any other id. Confirm the ids against openrouter.ai/models when changing them.
DEFAULT_OPENROUTER_MODELS = (
    "openai/gpt-4.1-mini",
    "anthropic/claude-sonnet-4.5",
    "google/gemini-2.5-flash",
    "qwen/qwen3-235b-a22b-2507",
)


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

    # Chat agent (the default provider). "off": chat disabled. "scripted": deterministic stand-in (tests, UI development).
    # "ollama": your own Ollama-compatible host given by llm_base_url / llm_model.
    llm_mode: Literal["off", "scripted", "ollama", "openrouter"] = "off"
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
    # Local debugging of the transformation operations: the response's diagnostics then also carry the
    # candidates that were rejected (which may contain expressions). Never logged, never saved,
    # development only.
    debug_compute: bool = False
    # OpenRouter (cloud). Off for users unless byok is true or a server key is set; Ollama stays the
    # local-first default and there is never an automatic fallback between providers.
    openrouter_model: str = ""
    openrouter_models: str = ""  # empty: DEFAULT_OPENROUTER_MODELS
    openrouter_byok: bool = False  # let users bring their own key from the chat UI
    openrouter_api_key: SecretStr = SecretStr("")  # operator key; mount from a Secret, never expose
    # Emails allowed to spend the operator key ("*" = every signed-in user: an explicit operator choice).
    openrouter_key_users: str = ""
    openrouter_daily_turns: int = Field(default=50, ge=1, le=100000)  # per user, operator key only
    openrouter_max_tokens: int = Field(default=4096, ge=64, le=32768)
    openrouter_data_collection: Literal["deny", "allow"] = "deny"
    openrouter_zdr: bool = False  # only route to zero-data-retention endpoints
    openrouter_timeout_seconds: float = Field(default=120.0, gt=0, le=900)
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

    @property
    def openrouter_model_list(self) -> list[str]:
        listed = [m.strip() for m in self.openrouter_models.split(",") if m.strip()]
        return listed or list(DEFAULT_OPENROUTER_MODELS)

    @property
    def openrouter_key_user_set(self) -> set[str]:
        return {u.strip().lower() for u in self.openrouter_key_users.split(",") if u.strip()}

    @property
    def has_server_key(self) -> bool:
        return bool(self.openrouter_api_key.get_secret_value())

    @property
    def ollama_configured(self) -> bool:
        return bool(self.llm_base_url and self.llm_model) and self.llm_mode in (
            "ollama",
            "openrouter",
        )

    @property
    def openrouter_available(self) -> bool:
        return self.llm_mode != "off" and (self.openrouter_byok or self.has_server_key)

    def validate_for_runtime(self) -> None:
        """Refuse to start in production with placeholder or weak secrets."""
        from .agent.openrouter import MODEL_ID

        for model in [self.openrouter_model, *self.openrouter_model_list]:
            if model and not MODEL_ID.match(model):
                raise RuntimeError(f"invalid OpenRouter model id: {model!r}")
        if self.llm_mode == "openrouter":
            if not self.openrouter_model:
                raise RuntimeError("MINERVA_LLM_MODE=openrouter needs MINERVA_OPENROUTER_MODEL")
            if not (self.openrouter_byok or self.has_server_key):
                raise RuntimeError(
                    "MINERVA_LLM_MODE=openrouter needs MINERVA_OPENROUTER_BYOK=true "
                    "or MINERVA_OPENROUTER_API_KEY"
                )
        if self.llm_mode == "ollama" and not (self.llm_base_url and self.llm_model):
            raise RuntimeError(
                "MINERVA_LLM_MODE=ollama needs MINERVA_LLM_BASE_URL and MINERVA_LLM_MODEL"
            )
        if self.llm_mode == "scripted" and self.env.lower() != "development":
            raise RuntimeError("the scripted stand-in model is for development only")
        if self.debug_compute and self.env.lower() != "development":
            raise RuntimeError("MINERVA_DEBUG_COMPUTE is for development only")
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
