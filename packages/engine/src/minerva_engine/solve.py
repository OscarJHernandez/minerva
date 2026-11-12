"""Polynomial equations: factorisation, closed forms where they exist, and what cannot be done.

For an equation `p(x) = 0` with rational coefficients the report states, per irreducible factor:
the degree, whether the roots can be written with radicals (always up to degree 4; decided by the
Galois group for degrees 5 and 6; not decided above), exact roots when SymPy can produce them, and
numerical roots to 30 digits. Every numerical root is substituted back, the root count must equal the
degree, and Vieta's relations must hold. SymPy cannot construct radicals for a solvable quintic: the
report then says a formula exists but Minerva cannot produce it. With symbolic coefficients only the
quadratic formula is given in closed form; for higher degree the report cites the general theory
(Cardano, Ferrari, Abel-Ruffini).
"""

from __future__ import annotations

from typing import Literal

import sympy as sp
from pydantic import BaseModel, Field

from .claims import split_equation
from .errors import MinervaMathError
from .latexparse import parse_latex
from .latexprint import to_latex
from .models import ExpressionInput, Limits, Status

MAX_DEGREE = 60
DIGITS = 30
TOL = sp.Float("1e-20")


class FactorReport(BaseModel):
    factor_latex: str
    multiplicity: int
    degree: int
    radicals: Literal["yes", "no", "undetermined"]
    radicals_reason: str
    exact_roots_latex: list[str] = Field(default_factory=list)


class RootReport(BaseModel):
    numeric: str
    real: bool
    multiplicity: int
    exact_latex: str | None = None


class SolveVerification(BaseModel):
    roots_substituted_back: bool = False
    max_residual: str = ""
    count_matches_degree: bool = False
    vieta_holds: bool = False
    exact_roots_match_numeric: bool | None = None  # None: no exact roots were produced


class SolveReport(BaseModel):
    status: Status
    equation_latex: str | None = None
    variable: str | None = None
    degree: int | None = None
    symbolic_coefficients: bool = False
    factors: list[FactorReport] = Field(default_factory=list)
    roots: list[RootReport] = Field(default_factory=list)
    verification: SolveVerification | None = None
    features: list[str] = Field(default_factory=list)  # problem features, used to retrieve methods
    notes: list[str] = Field(default_factory=list)
    message: str = ""
    duration_ms: int = 0


def _format(value: sp.Expr) -> str:
    """`-1.5`, or `0.5 + 1.25 i`, to about 25 significant digits."""
    re_part, im_part = sp.re(value), sp.im(value)
    if abs(im_part) < TOL:
        return str(sp.N(re_part, 25)).rstrip("0").rstrip(".") or "0"
    sign = "-" if im_part < 0 else "+"
    real_text = str(sp.N(re_part, 25)).rstrip("0").rstrip(".") or "0"
    imag_text = str(sp.N(abs(im_part), 25)).rstrip("0").rstrip(".")
    return f"{real_text} {sign} {imag_text} i"


def _fail(status: Status, message: str, latex: str | None = None) -> SolveReport:
    return SolveReport(status=status, message=message, equation_latex=latex)


def solve_polynomial(
    equation: str, variable: str | None, assumptions: dict[str, list[str]], limits: Limits
) -> SolveReport:
    sides = split_equation(equation)
    if sides is None or len(sides) != 2:
        return _fail(Status.PARSE_ERROR, "Write the equation as `left = right` (one equals sign).")
    try:
        lhs = parse_latex(sides[0], assumptions, limits)
        rhs = parse_latex(sides[1], assumptions, limits)
    except MinervaMathError as exc:
        return _fail(Status.PARSE_ERROR, str(exc))
    equation_latex = f"{to_latex(lhs)} = {to_latex(rhs)}"
    expr = sp.expand(lhs - rhs)
    symbols = sorted(expr.free_symbols, key=lambda s: s.name)
    if variable:
        x = next((s for s in symbols if s.name == variable), None)
        if x is None:
            return _fail(
                Status.PARSE_ERROR, f"The unknown {variable!r} does not occur.", equation_latex
            )
    elif len(symbols) == 1:
        x = symbols[0]
    elif not symbols:
        return _fail(Status.UNSUPPORTED, "There is no unknown in this equation.", equation_latex)
    else:
        names = ", ".join(s.name for s in symbols)
        return _fail(
            Status.UNSUPPORTED,
            f"Several symbols occur ({names}); say which one to solve for.",
            equation_latex,
        )

    try:
        poly = sp.Poly(expr, x)
    except sp.PolynomialError:
        return SolveReport(
            status=Status.UNSUPPORTED,
            equation_latex=equation_latex,
            variable=x.name,
            features=["solve:not-polynomial"],
            message=(
                "This is not a polynomial equation in the unknown, and Minerva only solves "
                "polynomial equations so far. A numerical root finder started from a guess "
                "(Newton's method, bisection on a bracket) is the usual approach."
            ),
        )
    degree = poly.degree()
    if degree < 1:
        return _fail(
            Status.UNSUPPORTED, "The equation does not depend on the unknown.", equation_latex
        )
    if degree > MAX_DEGREE:
        return _fail(
            Status.RESOURCE_LIMIT,
            f"Degree {degree} is above the limit of {MAX_DEGREE}.",
            equation_latex,
        )

    params = sorted(expr.free_symbols - {x}, key=lambda s: s.name)
    report = SolveReport(
        status=Status.OK,
        equation_latex=equation_latex,
        variable=x.name,
        degree=degree,
        symbolic_coefficients=bool(params),
        features=["poly", f"poly:degree:{min(degree, 7)}"],
    )
    if params:
        return _symbolic(report, poly, x, params)
    if not poly.domain.is_QQ and not poly.domain.is_ZZ:
        return _fail(
            Status.UNSUPPORTED,
            "Coefficients must be rational numbers (integers or fractions).",
            equation_latex,
        )
    return _numeric(report, poly, x)


def _symbolic(
    report: SolveReport, poly: sp.Poly, x: sp.Symbol, params: list[sp.Symbol]
) -> SolveReport:
    degree = report.degree or 0
    names = ", ".join(p.name for p in params)
    report.features.append("poly:symbolic-coefficients")
    report.verification = None
    if degree == 1:
        root = sp.solve(poly.as_expr(), x)[0]
        report.factors = [
            FactorReport(
                factor_latex=to_latex(poly.as_expr()),
                multiplicity=1,
                degree=1,
                radicals="yes",
                radicals_reason="linear",
                exact_roots_latex=[to_latex(root)],
            )
        ]
        report.message = (
            f"Linear in {x.name} with symbolic coefficients ({names}); the solution is exact."
        )
    elif degree == 2:
        roots = sp.solve(poly.as_expr(), x)
        report.factors = [
            FactorReport(
                factor_latex=to_latex(poly.as_expr()),
                multiplicity=1,
                degree=2,
                radicals="yes",
                radicals_reason="quadratic formula",
                exact_roots_latex=[to_latex(r) for r in roots],
            )
        ]
        report.message = (
            f"Quadratic with symbolic coefficients ({names}): the quadratic formula applies. The roots "
            "are exact; which are real depends on the sign of the discriminant."
        )
        report.features.append("poly:quadratic")
    elif degree in (3, 4):
        report.factors = [
            FactorReport(
                factor_latex=to_latex(poly.as_expr()),
                multiplicity=1,
                degree=degree,
                radicals="yes",
                radicals_reason="Cardano's formula" if degree == 3 else "Ferrari's method",
            )
        ]
        report.message = (
            f"A general formula in radicals exists for degree {degree} "
            f"({'Cardano' if degree == 3 else 'Ferrari'}), but with symbolic coefficients ({names}) it is "
            "very long and Minerva does not print it. Give numerical values for the coefficients to get "
            "the roots, or ask for the reduction to a depressed polynomial."
        )
        report.features.append("poly:general-formula-exists")
    else:
        report.factors = [
            FactorReport(
                factor_latex=to_latex(poly.as_expr()),
                multiplicity=1,
                degree=degree,
                radicals="no",
                radicals_reason="Abel-Ruffini: no general formula in radicals exists for degree 5 or higher",
            )
        ]
        report.message = (
            f"For a general polynomial of degree {degree} with symbolic coefficients ({names}) no formula "
            "in radicals exists (Abel-Ruffini theorem). Specific coefficient values may still be solvable; "
            "give numbers to analyse them, or solve numerically."
        )
        report.features.append("poly:no-general-formula")
    report.notes.append("No numerical roots: the coefficients are symbolic.")
    return report


def _radicals(factor: sp.Poly) -> tuple[Literal["yes", "no", "undetermined"], str]:
    d = factor.degree()
    if d <= 4:
        return "yes", {
            1: "linear",
            2: "quadratic formula",
            3: "Cardano's formula",
            4: "Ferrari's method",
        }[d]
    if d in (5, 6):
        try:
            group, _ = factor.galois_group()
        except (ValueError, NotImplementedError):
            return "undetermined", "SymPy could not compute the Galois group"
        order = int(group.order())
        if group.is_solvable:
            return "yes", f"the Galois group (order {order}) is solvable"
        return "no", f"the Galois group (order {order}) is not solvable (Galois's criterion)"
    return "undetermined", "solvability by radicals is only decided here up to degree 6"


def _numeric(report: SolveReport, poly: sp.Poly, x: sp.Symbol) -> SolveReport:
    degree = report.degree or 0
    _, factor_list = sp.factor_list(poly.as_expr(), x)
    root_reports: list[RootReport] = []
    exact_all: list[sp.Expr] = []
    any_unproduced_formula = False
    for factor_expr, mult in factor_list:
        factor = sp.Poly(factor_expr, x)
        if factor.degree() < 1:
            continue
        radicals, reason = _radicals(factor)
        exact: list[sp.Expr] = []
        if factor.degree() <= 4:
            found = sp.roots(factor, x, cubics=True, quartics=True)
            exact = [r for r, m in found.items() for _ in range(m)]
            if len(exact) != factor.degree():
                exact = []
        elif radicals == "yes":
            any_unproduced_formula = True
        report.factors.append(
            FactorReport(
                factor_latex=to_latex(factor.as_expr()),
                multiplicity=int(mult),
                degree=factor.degree(),
                radicals=radicals,
                radicals_reason=reason,
                exact_roots_latex=[to_latex(r) for r in exact],
            )
        )
        exact_all.extend(exact * int(mult))

    numeric = sp.Poly(poly.as_expr(), x).nroots(n=DIGITS, maxsteps=500)
    values = [sp.N(r, DIGITS) for r in numeric]
    exact_numeric = [sp.N(e, DIGITS) for e in exact_all]
    for value in sorted(
        values, key=lambda v: (bool(abs(sp.im(v)) > TOL), float(sp.re(v)), float(sp.im(v)))
    ):
        real = abs(sp.im(value)) < TOL
        match = next(
            (e for e in exact_all if abs(sp.N(e, DIGITS) - value) < sp.Float("1e-15")), None
        )
        root_reports.append(
            RootReport(
                numeric=_format(value),
                real=bool(real),
                multiplicity=1,
                exact_latex=to_latex(match) if match is not None else None,
            )
        )
    report.roots = root_reports

    # Verification: substitution, count, Vieta, and agreement of exact and numerical roots.
    coeffs = [sp.Rational(c) for c in poly.all_coeffs()]
    lead = coeffs[0]
    scale = max(abs(c) for c in coeffs)
    residual = max(abs(sp.N(poly.eval(v), 2 * DIGITS)) for v in values) if values else sp.Float(0)
    vieta_sum = abs(sum(values) + coeffs[1] / lead)
    vieta_prod = abs(sp.prod(values) - (-1) ** degree * coeffs[-1] / lead)
    report.verification = SolveVerification(
        roots_substituted_back=bool(residual < TOL * scale * degree),
        max_residual=sp.sstr(sp.N(residual, 3)),
        count_matches_degree=len(values) == degree,
        vieta_holds=bool(vieta_sum < TOL * scale and vieta_prod < TOL * scale),
        exact_roots_match_numeric=(
            _same_multiset(exact_numeric, values) if exact_numeric else None
        ),
    )
    v = report.verification
    ok = v.roots_substituted_back and v.count_matches_degree and v.vieta_holds
    if v.exact_roots_match_numeric is False:
        ok = False
    if not ok:
        report.status = Status.NOT_VERIFIED
        report.message = (
            "The computed roots failed Minerva's own checks, so they are not shown as results."
        )
        report.roots = []
        return report

    report.features.append("poly:numeric-roots")
    for f in report.factors:
        if (
            f.degree == 3
            and f.radicals == "yes"
            and sum(r.real for r in root_reports) == degree == 3
        ):
            report.notes.append(
                "A cubic with three real roots cannot be written with real radicals only: Cardano's "
                "formula passes through complex numbers (casus irreducibilis). The trigonometric form "
                "of the solution avoids this."
            )
            report.features.append("poly:casus-irreducibilis")
    if any(f.radicals == "no" for f in report.factors):
        report.features.append("poly:not-solvable-by-radicals")
    if any_unproduced_formula:
        report.notes.append(
            "A formula in radicals exists for at least one factor (its Galois group is solvable), "
            "but SymPy cannot construct it, so only numerical roots are given."
        )
    if any(f.degree >= 5 and f.radicals == "no" for f in report.factors):
        report.notes.append(
            "For the factor of degree 5 or higher no formula in radicals exists, so numerical methods "
            "(Newton's method, or the companion-matrix eigenvalues) are the way to compute roots."
        )
    nonsolvable = [f for f in report.factors if f.radicals == "undetermined"]
    if nonsolvable:
        report.notes.append("For degrees above 6 Minerva does not decide solvability by radicals.")
    report.message = (
        f"Degree {degree}: {len(root_reports)} roots found and verified by substitution "
        "and Vieta's relations."
    )
    return report


def _same_multiset(a: list[sp.Expr], b: list[sp.Expr]) -> bool:
    remaining = list(b)
    for e in a:
        index = next((i for i, v in enumerate(remaining) if abs(e - v) < sp.Float("1e-15")), None)
        if index is None:
            return False
        remaining.pop(index)
    return not remaining


def solve_request(data: ExpressionInput, variable: str | None, limits: Limits) -> SolveReport:
    return solve_polynomial(data.input, variable, data.assumptions, limits)
