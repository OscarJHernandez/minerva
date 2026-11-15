"""The sandbox children must not start one BLAS thread per core.

On a many-core Linux runner each OpenBLAS thread allocates a buffer, and under RLIMIT_AS that allocation
failed ("OpenBLAS error: Memory allocation still failed"), so every ODE run came back `resource_limit`.
"""

import os

from minerva_engine import sandbox


def test_the_forkserver_inherits_single_threaded_blas(monkeypatch) -> None:
    for name in sandbox._SINGLE_THREAD:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sandbox, "_ctx", None)
    sandbox._context()
    assert all(os.environ[name] == "1" for name in sandbox._SINGLE_THREAD)


def test_an_explicit_setting_is_respected(monkeypatch) -> None:
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "2")
    monkeypatch.setattr(sandbox, "_ctx", None)
    sandbox._context()
    assert os.environ["OPENBLAS_NUM_THREADS"] == "2"
