"""Vector calculus: typed vector expressions, exact lowering, independent numerical checks."""

from .compute import vector_calculus
from .models import (
    Blocked,
    Curve,
    FieldDecl,
    ModelCheck,
    Param,
    Premise,
    Region,
    Surface,
    VectorDiagnostics,
    VectorInput,
    VectorResult,
    VectorRule,
    VectorVerification,
)

__all__ = [
    "Blocked",
    "Curve",
    "FieldDecl",
    "ModelCheck",
    "Param",
    "Premise",
    "Region",
    "Surface",
    "VectorDiagnostics",
    "VectorInput",
    "VectorResult",
    "VectorRule",
    "VectorVerification",
    "vector_calculus",
]
