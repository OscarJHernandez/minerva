"""Vector calculus: typed vector expressions, exact lowering, independent numerical checks."""

from .compute import vector_calculus
from .models import (
    Blocked,
    FieldDecl,
    Premise,
    VectorDiagnostics,
    VectorInput,
    VectorResult,
    VectorRule,
    VectorVerification,
)

__all__ = [
    "Blocked",
    "FieldDecl",
    "Premise",
    "VectorDiagnostics",
    "VectorInput",
    "VectorResult",
    "VectorRule",
    "VectorVerification",
    "vector_calculus",
]
