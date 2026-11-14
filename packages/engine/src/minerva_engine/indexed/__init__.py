"""Finite sums, products and indexed variables, and the checks that make them trustworthy."""

from .guard import SUMS_REFUSAL, contains_indexed
from .randoms import Covariance, Expectation, Variance

__all__ = ["SUMS_REFUSAL", "Covariance", "Expectation", "Variance", "contains_indexed"]
