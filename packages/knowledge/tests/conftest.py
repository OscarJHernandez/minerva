import pytest
from minerva_engine import Engine, Limits
from minerva_knowledge import KnowledgeBase


@pytest.fixture(scope="session")
def kb() -> KnowledgeBase:
    return KnowledgeBase.load()


@pytest.fixture(scope="session")
def engine() -> Engine:
    return Engine(Limits(timeout_seconds=30, max_concurrency=2))
