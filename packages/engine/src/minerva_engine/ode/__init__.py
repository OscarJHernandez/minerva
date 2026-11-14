"""Ordinary differential equations: reading, structure and method suggestions (see analyze.py)."""

from .analyze import analyze
from .models import (
    MethodSuggestion,
    OdeAnalysis,
    OdeCondition,
    OdeDiagnostics,
    OdeInput,
    OdeInterpretation,
    OdeNumericInput,
    OdeNumericReport,
    OdeSolution,
    OdeSolutionItem,
    OdeStage,
    OdeStructure,
    OdeVerification,
)

__all__ = [
    "MethodSuggestion",
    "OdeAnalysis",
    "OdeCondition",
    "OdeDiagnostics",
    "OdeInput",
    "OdeInterpretation",
    "OdeNumericInput",
    "OdeNumericReport",
    "OdeSolution",
    "OdeSolutionItem",
    "OdeStage",
    "OdeStructure",
    "OdeVerification",
    "analyze",
]
