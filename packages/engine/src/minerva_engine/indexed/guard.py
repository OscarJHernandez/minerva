"""Sums, products, indexed variables and random-variable moments are refused by every pointwise check.

The numeric checkers draw one number per free symbol. For `Sum(x[i]**2, (i, 1, n))` that gives every
`x[i]` the same value, so `sum x_i^2 = (sum x_i)^2 / n` "agrees" at every sample (it is true for n = 1,
and for equal data). A random variable is sampled as a number too, which makes `E[XY] = E[X]E[Y]`
agree. None of that is evidence, so these objects only reach the instance-based checkers of the
`sums`, `expectation` and `statistical_model` operations.
"""

from __future__ import annotations

import sympy as sp

from .randoms import RANDOM_TYPES

SUMS_REFUSAL = (
    "sums, products, indexed variables and expectations are available only in the sums, "
    "expectation and statistics operations"
)

_TYPES = (sp.Sum, sp.Product, sp.Indexed, sp.IndexedBase, *RANDOM_TYPES)


def contains_indexed(expr: sp.Basic) -> bool:
    """Does `expr` contain a sum, product, indexed variable or expectation?"""
    return bool(expr.has(*_TYPES))
