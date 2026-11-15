"""The `second` action: quadratic form, its verification, and diagnostics along a path."""

from __future__ import annotations

from typing import Any, Literal

import mpmath as mp

from ..errors import ParseError
from ..models import Status
from .analysis import legendre_matrix
from .classify import SecondEvidence, classify
from .firstvar import first_variation
from .models import (
    Diagnostics,
    DirectionResult,
    LegendreSamples,
    RitzResult,
    SecondNumeric,
    SecondVariation,
    VariationInput,
    VariationReport,
)
from .numeric_setup import path_field, prepare
from .numerics import JetField, Numerics, ShiftedField
from .quadrature import fd_table
from .report import checks_out
from .secondvar import (
    STEPS2,
    QuadraticForm,
    admissible_basis,
    check_second_variation,
    quadratic_expression,
    quadratic_value,
    ritz_inertia,
    second_form,
)
from .setup import Problem
from .userverify import _direct_claim, _fixed_slots, _poly
from .verify import action_value

STATIONARY_TOL = mp.mpf("1e-8")
ZERO_REL = mp.mpf("1e-9")


def second(problem: Problem, data: VariationInput, deadline: float) -> VariationReport:
    form = second_form(problem)
    check = check_second_variation(problem, form, deadline=deadline)
    if not check.passed:
        return VariationReport(
            status=Status.NOT_VERIFIED,
            action="second",
            message="The second variation did not pass its independent finite-difference "
            "check, so it is not shown.",
        )
    jets = problem.jets
    lo, hi = problem.latex(problem.lower), problem.latex(problem.upper)
    t = problem.variable.name
    interior = problem.latex(quadratic_expression(jets, form.interior))
    endpoint = ""
    for end, point in (("upper", hi), ("lower", lo)):
        entries = form.endpoint[end]  # type: ignore[index]
        if entries:
            endpoint += rf" + \left.\left({problem.latex(quadratic_expression(jets, entries))}\right)\right|_{{{t}={point}}}"
    matrix, _ = legendre_matrix(problem)
    legendre = (
        r"\begin{pmatrix}"
        + r" \\ ".join(
            " & ".join(problem.latex(matrix[r, c]) for c in range(matrix.cols))
            for r in range(matrix.rows)
        )
        + r"\end{pmatrix}"
        if matrix.rows
        else ""
    )
    numeric = _numeric(problem, data, form, deadline) if data.paths else None
    result = SecondVariation(
        convention_latex=r"S[x+\varepsilon\eta]=S[x]+\varepsilon\,\delta S+\tfrac12\varepsilon^{2}\,\delta^{2}S+\cdots",
        quadratic_form_latex=rf"\delta^{{2}}S=\int_{{{lo}}}^{{{hi}}}\left({interior}\right)\mathrm{{d}}{t}{endpoint}",
        endpoint_form_latex=endpoint.removeprefix(" + "),
        legendre_matrix_latex=legendre,
        checks=checks_out(
            problem,
            check,
            "Numerical evidence: second differences of the action along polynomial perturbations "
            "agree with the claimed quadratic form. It is not a proof.",
        ),
        numeric=numeric,
    )
    return VariationReport(
        status=Status.OK,
        action="second",
        second=result,
        message="Second variation derived and confirmed by second differences of the action.",
        diagnostics=Diagnostics(stages=len(check.comparisons)),
    )


def _numeric(
    problem: Problem, data: VariationInput, form: QuadraticForm, deadline: float
) -> SecondNumeric:
    if len(data.paths) != 1:
        raise ParseError("give exactly one path for the second-variation diagnostics")
    num = prepare(problem, data, seed=20260502)
    depth = max(problem.jets.orders)
    field, _ = path_field(num, problem, data.paths[0], data, depth=depth)
    return diagnostics(problem, num, field, form)


def diagnostics(
    problem: Problem, num: Numerics, field: JetField, form: QuadraticForm
) -> SecondNumeric:
    """Stationarity, Legendre samples, Ritz inertia and certified directions along a path."""
    jets = num.jets
    depth = max(jets.orders)
    real = _is_real(num)
    derived = first_variation(problem)

    # 1. is the path stationary (along the admissible perturbations that were tried)?
    claim = _direct_claim(num, derived, None)
    degree = max(2 * depth + 3, len(problem.conditions) + 4)
    worst = mp.mpf(0)
    floor = mp.mpf("1e-12") * (1 + abs(action_value(num, field)[0]))
    for kind, only in (("bubble", None), ("free_end", None), ("free_end", None)):
        eta = num.field(_poly(num, kind, only, degree))
        value, scale, _ = claim(field, eta)
        worst = max(worst, abs(value) / (scale + floor))
    stationary: Literal["consistent", "violated"] = (
        "consistent" if worst <= STATIONARY_TOL else "violated"
    )

    # 2. Legendre matrix along the path
    legendre = None
    negative_w = positive_w = False
    matrix, rows = legendre_matrix(problem)
    if rows and real:
        compiled = [
            [num.compile(matrix[r, c]) for c in range(matrix.cols)] for r in range(matrix.rows)
        ]
        lows, highs = [], []
        for n in range(9):
            s = num.lower + (num.upper - num.lower) * n / 8
            values = num.jet_values(s, field)
            w = mp.matrix([[mp.re(fn(values)) for fn in row] for row in compiled])
            eig = mp.eigsy((w + w.T) / 2, eigvals_only=True)
            lows.append(min(eig))
            highs.append(max(eig))
        low, high = min(lows), max(highs)
        tol = ZERO_REL * max(abs(low), abs(high), mp.mpf(1))
        negative_w, positive_w = low < -tol, high > tol
        legendre = LegendreSamples(
            points=9, smallest_eigenvalue=float(low), largest_eigenvalue=float(high)
        )

    # 3. Ritz restriction to admissible polynomial perturbations
    directions: list[DirectionResult] = []
    ritz_out = None
    negative_cert = positive_cert = False
    if real:
        vanishing = {i: _fixed_slots(problem, i) for i in range(jets.count)}
        per = max(3, min(6, 12 // jets.count))
        basis = admissible_basis(num, per, {i: list(v) for i, v in vanishing.items()})
        ritz = ritz_inertia(num, form, field, basis)
        ritz_out = RitzResult(
            dimension=len(ritz.eigenvalues),
            eigenvalues=[float(v) for v in ritz.eigenvalues[:12]],
            negative=ritz.negative,
            zero=ritz.zero,
            positive=ritz.positive,
        )
        if stationary == "consistent":
            for _value, (eta_poly,) in ritz.directions[:2]:
                result = _certify(
                    num, form, field, num.field(eta_poly), "negative", "most negative direction"
                )
                directions.append(result)
                negative_cert = negative_cert or result.certified
            if ritz.positive:
                top = _top_direction(num, form, field, basis)
                if top is not None:
                    result = _certify(
                        num, form, field, num.field(top), "positive", "most positive direction"
                    )
                    directions.append(result)
                    positive_cert = result.certified
    facts, missing = classify(
        SecondEvidence(
            real_valued=real,
            stationary=stationary,
            negative_certified=negative_cert,
            positive_certified=positive_cert,
            legendre_negative=negative_w,
            legendre_positive=positive_w,
        )
    )
    return SecondNumeric(
        stationary=stationary,
        stationary_residual=float(worst),
        legendre=legendre,
        directions=directions,
        ritz=ritz_out,
        facts=facts,
        not_established=missing,
    )


def _is_real(num: Numerics) -> bool:
    return all(abs(mp.im(v)) == 0 for v in num.params.values())


def _top_direction(num: Numerics, form: QuadraticForm, field: JetField, basis: Any) -> Any:
    # Largest eigenvalue direction: negate nothing, take the opposite end of the same spectrum.
    ritz = ritz_inertia(num, _negated(form), field, basis)
    if not ritz.directions:
        return None
    return ritz.directions[0][1][0]


def _negated(form: QuadraticForm) -> QuadraticForm:
    return QuadraticForm(
        {k: -v for k, v in form.interior.items()},
        {end: {k: -v for k, v in entries.items()} for end, entries in form.endpoint.items()},
    )


def _certify(
    num: Numerics, form: QuadraticForm, path: JetField, eta: JetField, expect: str, label: str
) -> DirectionResult:
    """δ²S along `eta` by two quadratures, a second difference and the measured change of S."""
    value, scale, err = quadratic_value(num, form, path, eta)
    fast_ok = err <= mp.mpf("1e-12") * (1 + abs(value))
    sign = "negative" if value < 0 else ("positive" if value > 0 else "zero")
    errors: list[mp.mpf] = [err]

    def action(e: mp.mpf) -> Any:
        total, quad = action_value(num, ShiftedField(path, eta, e))
        errors.append(quad)
        return total

    floor = mp.mpf("1e-12") * (1 + abs(action_value(num, path)[0]))
    table = fd_table(action, value, scale, STEPS2, order=2, noise=max(errors), floor=floor)
    base = action_value(num, path)[0]
    changes = [mp.re(action(e) - base) for e in (mp.mpf("1e-2"), mp.mpf("-1e-2"), mp.mpf("5e-3"))]
    measured = all((c < 0) == (expect == "negative") and c != 0 for c in changes)
    certified = bool(fast_ok and table.verdict == "agree" and measured and sign == expect)
    return DirectionResult(
        label=label,
        second_variation=mp.nstr(value, 12),
        sign=sign,
        certified=certified,
    )
