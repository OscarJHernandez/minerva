"""What every reference problem is made of: a problem, and the verdict on an outcome."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .service import Engine

Judge = Callable[[Engine], tuple[str, str]]  # -> (outcome, detail)


@dataclass(frozen=True)
class Problem:
    id: str
    category: str
    source: str  # where the expected outcome comes from, by locator only
    expected: str  # "correct" or "refused"
    judge: Judge


def _verdict(expected: str, ok: bool, agrees: bool | None, detail: str) -> tuple[str, str]:
    """`ok`: the engine returned a result; `agrees`: that result matches the oracle (None: not tested)."""
    if expected == "refused":
        return (
            ("wrong", f"returned a result where a refusal was due: {detail}")
            if ok
            else (
                "refused_as_expected",
                detail,
            )
        )
    if not ok:
        return "unsupported_regression", detail
    return ("correct", detail) if agrees else ("wrong", detail)
