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


def test_an_exception_in_the_child_reports_its_class_and_location_never_its_message() -> None:
    from minerva_engine import Limits

    secret = "secret@example.com"
    result = sandbox.Sandbox(Limits(timeout_seconds=20)).run("simplify", {secret: 1})
    assert result.outcome is sandbox.Outcome.ERROR and result.value is None
    assert result.error_type == "KeyError"
    # the innermost frame of OUR code that raised it, not the sandbox's own plumbing
    assert result.error_module == "minerva_engine.ops"
    assert isinstance(result.error_line, int) and result.error_line > 0
    assert secret not in repr(result)


def test_a_class_that_is_not_ours_is_reported_as_other() -> None:
    class Custom(Exception):
        pass

    Custom.__module__ = "somewhere_else"
    name, module, line = sandbox._site(Custom("boom"))
    assert name == "other" and module is None and line is None
