"""Independent reference problems: answers established without Minerva's own result path.

One module per mathematical domain; each exposes `PROBLEMS`. `core.py` holds the problem type, the verdict rules
and the helpers shared by the judges. `minerva_engine.reference_benchmark` runs them all and writes the report.
"""

from __future__ import annotations

from . import (
    algebra,
    calculus,
    distribution,
    matrix,
    mechanics,
    series,
    statmech,
    substitution,
    variational,
)
from .core import Problem

PROBLEMS: list[Problem] = [
    *matrix.PROBLEMS,
    *distribution.PROBLEMS,
    *substitution.PROBLEMS,
    *mechanics.PROBLEMS,
    *calculus.PROBLEMS,
    *series.PROBLEMS,
    *algebra.PROBLEMS,
    *statmech.PROBLEMS,
    *variational.PROBLEMS,
]

__all__ = ["PROBLEMS", "Problem"]
