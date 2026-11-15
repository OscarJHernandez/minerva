"""Verification of a telescoping certificate, on plain data.

`verify` takes the integrand, the operator and `G` as expressions, so a test can hand it a
forged certificate. The proof is exact: `(L f - dG/dx) / f` must simplify to zero. Two numerical
legs that share nothing with that simplification are reported next to it and may only veto: a
numerical disagreement rejects the certificate, numerical agreement never accepts it.
"""

from __future__ import annotations

import sympy as sp

from ..differentiate import check_derivative
from ..verify import numeric_detail
from .models import CertificateCheck
from .operators import DifferentialOperator

SIZE_CAP = 4000  # nodes: beyond this the exact check is skipped and the certificate is rejected


def _zero(expr: sp.Expr) -> bool:
    """Exact zero test for a ratio that is a rational function of x for hyperexponential f."""
    expr = sp.powsimp(sp.expand(expr))
    if expr == 0:
        return True
    try:
        return bool(sp.simplify(expr) == 0)
    except Exception:
        return False


def verify(
    f: sp.Expr, x: sp.Symbol, operator: DifferentialOperator, certificate: sp.Expr
) -> CertificateCheck:
    if operator.is_zero:
        return CertificateCheck(note="the operator is zero: a trivial relation says nothing")
    lf = operator.apply(f)
    gx = sp.diff(certificate, x)
    if (
        sum(1 for _ in sp.preorder_traversal(lf)) + sum(1 for _ in sp.preorder_traversal(gx))
        > SIZE_CAP
    ):
        return CertificateCheck(note="the expressions are too large to verify exactly")
    exact = _zero((lf - gx) / f)
    checked, agreed, _ = numeric_detail(lf, gx)
    derivative = check_derivative(certificate, x, lf)
    numeric_ok = checked == agreed and derivative.points_checked == derivative.points_agreed
    passed = exact and numeric_ok
    if not exact:
        note = "L f - dG/dx does not simplify to zero"
    elif not numeric_ok:
        note = (
            "the exact check passed but a numerical leg disagrees, so the certificate is rejected"
        )
    elif checked == 0:
        note = "exact zero; no sample point was usable for the numerical legs"
    else:
        note = "L f - dG/dx simplifies to zero; both numerical legs agree"
    return CertificateCheck(
        symbolic_zero=exact,
        numeric_points_checked=checked,
        numeric_points_agreed=agreed,
        derivative_points_checked=derivative.points_checked,
        derivative_points_agreed=derivative.points_agreed,
        passed=passed,
        note=note,
    )
