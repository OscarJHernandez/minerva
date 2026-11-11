import os
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from minerva_api.db import Base, make_engine
from minerva_api.main import create_app
from minerva_api.settings import Settings
from pydantic import SecretStr

PASSWORD = "correct horse battery"


def make_settings(**over: object) -> Settings:
    base: dict[str, object] = {
        "env": "development",
        "database_url": os.environ.get("MINERVA_TEST_DATABASE_URL")
        or "sqlite+pysqlite:///:memory:",
        "session_secret": SecretStr("test-secret-value-test-secret-value!"),
        "compute_timeout_seconds": 30.0,
        "compute_max_concurrency": 2,
        "auth_rate_per_minute": 1000,
        "compute_rate_per_minute": 1000,
    }
    base.update(over)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type]  # hermetic: ignore .env


def build_app(**over: object):  # type: ignore[no-untyped-def]
    """Fresh app on a clean database (SQLite in memory, or Postgres if configured)."""
    settings = make_settings(**over)
    if not settings.database_url.startswith("sqlite"):
        engine = make_engine(settings.database_url)
        Base.metadata.drop_all(engine)
        engine.dispose()
    return create_app(settings, create_tables=True)


@pytest.fixture
def app():  # type: ignore[no-untyped-def]
    return build_app()


@pytest.fixture
def client(app) -> Iterator[TestClient]:  # type: ignore[no-untyped-def]
    with TestClient(app) as c:
        yield c


def register(client: TestClient, email: str = "ada@example.com") -> None:
    r = client.post("/v1/auth/register", json={"email": email, "password": PASSWORD})
    assert r.status_code == 201, r.text


@pytest.fixture
def authed(client: TestClient) -> TestClient:
    register(client)
    return client
