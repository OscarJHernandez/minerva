"""The problem type, the verdict rules and helpers shared by the domain modules.

A problem expects either a **correct** answer (the engine's result must agree with the oracle) or a **refusal**
(the engine must not return an `ok` result: the oracle shows that any such result would be wrong or unjustified).

Fields beyond the id, domain and source:
  oracle    what the expected answer comes from: `mpmath_quad`, `mpmath_diff`, `mpmath_taylor`,
            `mpmath_polyroots`, `mpmath_expm`, `mpmath_odefun_taylor`, `closed_form`, `arithmetic`, `exact_bracket`, `exact_solution`,
            `finite_difference`. Never Minerva's own computation.
  evidence  the strongest kind of check the *engine itself* reports for an accepted result: `symbolic_proof`
            (a symbolic zero test passed), `numeric_agreement` (independent numerical checks only),
            `structural` (substitution back, Vieta, bracket, a rule's conditions), `conditional` (holds given
            premises the user supplied) or `refusal` (the right answer is no answer). Numerical agreement is
            never called a proof.
  tier      `pr` problems run in every pull request; `extended` ones run in the scheduled tier.
  pending   a defect or capability gap id. A pending problem is *expected not to pass yet*: its outcome is
            `pending`, and if it starts to pass the outcome is `pending_now_passes`, which fails the suite until
            the problem is promoted (pending removed). It is never counted as correct.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mpmath as mp
import sympy as sp

from ..models import ExpressionInput
from ..service import Engine
from ..textparse import parse_text

mp.mp.dps = 30

Judge = Callable[[Engine], tuple[str, str]]  # -> (outcome, detail)

ORACLES = frozenset(
    {
        "mpmath_quad",
        "mpmath_diff",
        "mpmath_taylor",
        "mpmath_polyroots",
        "mpmath_expm",
        "mpmath_odefun_taylor",
        "closed_form",
        "arithmetic",
        "exact_bracket",
        "exact_solution",
        "finite_difference",
    }
)
EVIDENCE = frozenset(
    {"symbolic_proof", "numeric_agreement", "structural", "conditional", "refusal"}
)
TIERS = ("pr", "extended")


@dataclass(frozen=True)
class Problem:
    id: str
    category: str
    source: str  # where the expected outcome comes from, by locator only (no quoted text)
    expected: str  # "correct" or "refused"
    judge: Judge
    oracle: str = "closed_form"
    evidence: str = "numeric_agreement"
    tier: str = "pr"
    pending: str | None = None

    def __post_init__(self) -> None:
        assert self.expected in ("correct", "refused"), self.id  # noqa: S101
        assert self.oracle in ORACLES, (self.id, self.oracle)  # noqa: S101
        assert self.evidence in EVIDENCE, (self.id, self.evidence)  # noqa: S101
        assert self.tier in TIERS, (self.id, self.tier)  # noqa: S101
        assert (self.expected == "refused") == (self.evidence == "refusal"), self.id  # noqa: S101


def expr(latex: str, **assumptions: list[str]) -> ExpressionInput:
    return ExpressionInput(input=latex, assumptions=dict(assumptions))


def text(source: str, **assumptions: list[str]) -> ExpressionInput:
    return ExpressionInput(input=source, format="text", assumptions=dict(assumptions))


def verdict(expected: str, ok: bool, agrees: bool | None, detail: str) -> tuple[str, str]:
    """`ok`: the engine returned a result; `agrees`: that result matches the oracle (None: not tested)."""
    if expected == "refused":
        return (
            ("wrong", f"returned a result where a refusal was due: {detail}")
            if ok
            else ("refused_as_expected", detail)
        )
    if not ok:
        return "unsupported_regression", detail
    return ("correct", detail) if agrees else ("wrong", detail)


def mp_matrix(entries: list[list[int]]) -> mp.matrix:
    return mp.matrix(entries)


def close(a: mp.matrix, b: mp.matrix, tol: float = 1e-12) -> bool:
    return bool(mp.norm(a - b) <= tol * max(1, mp.norm(b)))


def numeric_entries(entries: list[list[str]], **values: float) -> mp.matrix:
    syms = {k: sp.Symbol(k) for k in values}
    rows = []
    for row in entries:
        out = []
        for cell in row:
            e = parse_text(cell) if isinstance(cell, str) else sp.sympify(cell)
            out.append(
                mp.mpmathify(complex(e.subs({syms[k]: v for k, v in values.items()}).evalf(30)))
            )
        rows.append(out)
    return mp.matrix(rows)


def value_of(source: str, **at: float) -> mp.mpf:
    """The number a text expression has at the given values (parsed by Minerva's parser, evaluated by SymPy)."""
    e = parse_text(source)
    return mp.mpmathify(complex(e.subs({sp.Symbol(k): v for k, v in at.items()}).evalf(30)))
