"""Coordinate-based differential geometry: metrics, connection, curvature, geodesics."""

from .geometry import geometry
from .models import (
    HARD_MAX_DIMENSION,
    MAX_DIMENSION,
    QUANTITIES,
    Check,
    Component,
    FunctionDecl,
    GeometryInput,
    GeometryResult,
    GeometryVerification,
    QuantityResult,
    StageDiagnostic,
    TensorField,
)

__all__ = [
    "HARD_MAX_DIMENSION",
    "MAX_DIMENSION",
    "QUANTITIES",
    "Check",
    "Component",
    "FunctionDecl",
    "GeometryInput",
    "GeometryResult",
    "GeometryVerification",
    "QuantityResult",
    "StageDiagnostic",
    "TensorField",
    "geometry",
]
