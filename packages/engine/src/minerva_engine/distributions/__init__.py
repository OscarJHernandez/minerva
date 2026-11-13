"""Verified calculus of Dirac delta and Heaviside distributions (see ``guard`` for the reasons)."""

from .guard import DISTRIBUTION_REFUSAL, contains_distribution

__all__ = ["DISTRIBUTION_REFUSAL", "contains_distribution"]
