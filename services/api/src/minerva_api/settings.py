"""Runtime configuration, read from MINERVA_* environment variables."""

from __future__ import annotations

from functools import lru_cache

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

    llm_base_url: str = ""  # reserved for the optional LLM feature; unused in this milestone
    llm_model: str = ""

    @field_validator("allowed_origins")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()

    @property
    def origins(self) -> list[str]:
        return [o.strip().rstrip("/") for o in self.allowed_origins.split(",") if o.strip()]

    def validate_for_runtime(self) -> None:
        """Refuse to start in production with placeholder or weak secrets."""
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
