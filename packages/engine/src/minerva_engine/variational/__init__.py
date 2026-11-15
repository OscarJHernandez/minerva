"""General variational calculus: functionals of several functions and their derivatives.

First variation with explicit integration by parts, Euler-Lagrange equations of any order,
complete boundary forms, numerical action evaluation, finite-difference verification, numerical
stationary paths and second variations. See `docs/variational-calculus.md`.
"""

from .models import (
    Analysis,
    EndpointCondition,
    FunctionalInput,
    VariationInput,
    VariationReport,
)
from .operation import variation

__all__ = [
    "Analysis",
    "EndpointCondition",
    "FunctionalInput",
    "VariationInput",
    "VariationReport",
    "variation",
]
