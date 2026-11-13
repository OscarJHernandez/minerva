"""Detecting distributions, so that pointwise machinery never touches them.

A Dirac delta is 0 at every point a numerical sampler can draw, and ``Heaviside`` differs from
``(1 + sign)/2`` only at one point. Sampling therefore "verifies" wrong delta identities
(``0 = 5*DiracDelta(x)`` agrees at every sample). Every operation except ``distribution`` refuses
such input, and ``verify_equal`` refuses to pass it.
"""

from __future__ import annotations

import sympy as sp

DISTRIBUTION_REFUSAL = (
    "This expression contains a Dirac delta or Heaviside function. These are distributions, not "
    "ordinary functions, so this operation cannot check results involving them by sampling. "
    "Use the distribution operation instead."
)


def contains_distribution(expr: sp.Basic) -> bool:
    """True if ``expr`` contains ``DiracDelta`` or ``Heaviside`` anywhere."""
    return bool(expr.has(sp.DiracDelta, sp.Heaviside))
