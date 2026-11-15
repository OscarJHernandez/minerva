"""Approximation of expressions and functions: series, rational and polynomial approximants, sums."""

from .models import ApproximationInput, ApproximationReport
from .operation import approximation

__all__ = ["ApproximationInput", "ApproximationReport", "approximation"]
