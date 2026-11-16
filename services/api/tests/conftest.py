import os
import re
from collections.abc import Iterator
from functools import cache

import pytest
from fastapi.testclient import TestClient
from minerva_api.db import Base, make_engine
from minerva_api.main import create_app
from minerva_api.settings import Settings
from pydantic import SecretStr
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

PASSWORD = "correct horse battery"
WORKER = re.compile(r"gw[0-9]+")


@cache
def _worker_database(url: str, worker: str) -> str:
    """`url` with the database name suffixed by the xdist worker (`..._gw0`), created if it does not exist.

    Every test drops and recreates the tables of its database, so two workers must not share one. The
    worker id is validated and the name quoted by the dialect, never interpolated raw.
    """
    if not WORKER.fullmatch(worker):
        raise RuntimeError("unexpected pytest-xdist worker id")
    parsed = make_url(url)
    name = f"{parsed.database}_{worker}"
    admin = create_engine(url, isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": name}
            ).scalar()
            if not exists:
                conn.execute(
                    text(f"CREATE DATABASE {admin.dialect.identifier_preparer.quote(name)}")
                )
    finally:
        admin.dispose()
    return parsed.set(database=name).render_as_string(hide_password=False)


def database_url_for_tests() -> str:
    """The database of this test process: the configured one, or a private one per xdist worker.

    SQLite in memory is already private to each process. A PostgreSQL URL gets one database per worker so the
    API tests can run in parallel (`pytest -n 3`); without xdist the configured database is used as before.
    """
    url = os.environ.get("MINERVA_TEST_DATABASE_URL") or "sqlite+pysqlite:///:memory:"
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if url.startswith("sqlite") or not worker:
        return url
    return _worker_database(url, worker)


def make_settings(**over: object) -> Settings:
    base: dict[str, object] = {
        "env": "development",
        "database_url": database_url_for_tests(),
        "session_secret": SecretStr("test-secret-value-test-secret-value!"),
        # A wall-clock cap for the test app, not a mathematical tolerance. The conserved-quantity search
        # takes ~7 s alone and 20-40 s on a loaded CI runner, which a 30 s cap turned into
        # `resource_limit`. Tests of the timeout pass their own small value (see `build_app(...)`).
        "compute_timeout_seconds": 90.0,
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
