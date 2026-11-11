import pytest

from minerva_engine import Engine, Limits


@pytest.fixture(scope="session")
def engine() -> Engine:
    return Engine(Limits(timeout_seconds=20, max_concurrency=2))
