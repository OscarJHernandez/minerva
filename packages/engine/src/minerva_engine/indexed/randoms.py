"""Expectation, variance and covariance as unevaluated SymPy functions.

`sympy.stats.symbolic_probability` was measured and not used as the container: it needs `RandomSymbol`
objects bound to a probability space, leaves `Cov(X, X)` unreduced, does not pull constants out of a nested
expectation, and has no notion of declared independence. These three classes never evaluate by
themselves; `indexed.expectation` rewrites them by explicit rules and the result is checked against
explicit finite joint distributions.
"""

from __future__ import annotations

import sympy as sp


class Expectation(sp.Function):  # type: ignore[misc]
    """E[a]: the expected value of the random expression `a`."""

    nargs = 1

    def _eval_is_real(self) -> bool:
        return True


class Variance(sp.Function):  # type: ignore[misc]
    """Var(a)."""

    nargs = 1

    def _eval_is_real(self) -> bool:
        return True

    def _eval_is_nonnegative(self) -> bool:
        return True


class Covariance(sp.Function):  # type: ignore[misc]
    """Cov(a, b); symmetric, so the arguments are kept in a canonical order."""

    nargs = 2

    @classmethod
    def eval(cls, a: sp.Expr, b: sp.Expr) -> sp.Expr | None:
        if sp.default_sort_key(a) > sp.default_sort_key(b):
            return cls(b, a)
        return None

    def _eval_is_real(self) -> bool:
        return True


RANDOM_TYPES = (Expectation, Variance, Covariance)
