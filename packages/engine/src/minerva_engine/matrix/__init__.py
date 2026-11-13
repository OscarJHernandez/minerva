"""Noncommutative matrix calculus: exponentials, derivatives, integrals (see calculus.py)."""

from .calculus import matrix_calculus
from .models import (
    MatrixAlternative,
    MatrixCalculusInput,
    MatrixCalculusResult,
    MatrixDecl,
    MatrixDerivative,
    MatrixDiagnostics,
    MatrixStage,
    MatrixStep,
    MatrixVerification,
)

__all__ = [
    "MatrixAlternative",
    "MatrixCalculusInput",
    "MatrixCalculusResult",
    "MatrixDecl",
    "MatrixDerivative",
    "MatrixDiagnostics",
    "MatrixStage",
    "MatrixStep",
    "MatrixVerification",
    "matrix_calculus",
]
