"""Noncommutative matrix calculus: exponentials, derivatives, integrals (see calculus.py)."""

from .calculus import matrix_calculus
from .models import (
    MatrixCalculusInput,
    MatrixCalculusResult,
    MatrixDecl,
    MatrixDiagnostics,
    MatrixStage,
    MatrixStep,
    MatrixVerification,
)

__all__ = [
    "MatrixCalculusInput",
    "MatrixCalculusResult",
    "MatrixDecl",
    "MatrixDiagnostics",
    "MatrixStage",
    "MatrixStep",
    "MatrixVerification",
    "matrix_calculus",
]
