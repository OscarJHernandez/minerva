"""Perturbation expansions: against textbook and closed-form references, wrong expansions rejected, and
the cases where no small parameter or no periodic structure means nothing is shown."""

from __future__ import annotations

from typing import Any

import mpmath as mp
import pytest
import sympy as sp
from minerva_engine import Engine, Limits, OdeSolveInput, Status
from minerva_engine.ode import OdeSolution
from minerva_engine.ode.approximation import _check_expansion
from minerva_engine.ode.models import OdeNumericInput
from minerva_engine.ode.numerical import solve_numerical
from minerva_engine.ode.reading import read_ode
from minerva_engine.ode.solve import solve_symbolic

L = Limits(timeout_seconds=60)


def perturb(*equations: str, order: int = 2, **kw: Any) -> OdeSolution:
    request = OdeSolveInput(equations=list(equations), method="perturbation", order=order, **kw)
    return solve_symbolic(request, L)


def expansion(solution: OdeSolution, variable: str = "t", small: str = "epsilon") -> sp.Expr:
    return sp.sympify(
        solution.solutions[0].rhs_text[0],
        locals={variable: sp.Symbol(variable, real=True), small: sp.Symbol(small)},
    )


GROWTH = {
    "equations": [r"y' = -y + \epsilon y^2"],
    "independent": "t",
    "conditions": ["y(0) = 1"],
    "small_parameter": "epsilon",
}


def test_a_regular_expansion_matches_the_taylor_expansion_of_the_exact_solution() -> None:
    # y' + y = eps y^2, y(0) = 1 is a Bernoulli equation: y = 1 / (eps + (1 - eps) exp(t)).
    r = perturb(
        *GROWTH["equations"], order=3, **{k: v for k, v in GROWTH.items() if k != "equations"}
    )
    assert r.status is Status.OK, r.message
    (item,) = r.solutions
    assert item.kind == "perturbation" and item.expansion is not None
    assert item.expansion.kind == "regular_perturbation" and item.expansion.order == 3
    assert item.expansion.secular_orders == []
    t, eps = sp.Symbol("t", real=True), sp.Symbol("epsilon")
    poly = expansion(r)
    for value in (0.5, 1.3, 2.4):
        exact = lambda e, v=value: 1 / (e + (1 - e) * mp.exp(v))  # noqa: E731
        reference = [float(c) for c in mp.taylor(exact, 0, 3)]
        got = [float(poly.subs(t, value).coeff(eps, k)) for k in range(4)]
        assert got == pytest.approx(reference, rel=1e-9, abs=1e-12)
    v = item.verification
    assert v.passed and v.points_checked >= 4 and v.points_agreed == v.points_checked
    assert all(c.passed for c in item.expansion.checks)
    assert all(c.satisfied for c in item.conditions)
    assert "O\\left(\\epsilon^{4}\\right)" in item.equations_latex[0]


def test_the_first_order_correction_of_the_duffing_equation_is_the_textbook_one() -> None:
    # y'' + y + eps y^3 = 0, y(0) = 1, y'(0) = 0:  y1 = (cos 3t - cos t)/32 - (3/8) t sin t
    r = perturb(
        r"y'' + y + \epsilon y^3 = 0",
        order=1,
        independent="t",
        conditions=["y(0) = 1", "y'(0) = 0"],
        small_parameter="epsilon",
    )
    assert r.status is Status.OK, r.message
    t = sp.Symbol("t", real=True)
    poly = sp.sympify(r.solutions[0].rhs_text[0], locals={"t": t, "epsilon": sp.Symbol("epsilon")})
    first = poly.coeff(sp.Symbol("epsilon"), 1)
    for value in (0.7, 2.1):
        want = (mp.cos(3 * value) - mp.cos(value)) / 32 - mp.mpf(3) / 8 * value * mp.sin(value)
        assert float(first.subs(t, value)) == pytest.approx(float(want), rel=1e-9)
    # the t sin t term is secular: the expansion must say so
    assert r.solutions[0].expansion is not None and r.solutions[0].expansion.secular_orders == [1]
    assert any("Secular" in n for n in r.notes)
    boundedness = [c for c in r.solutions[0].expansion.checks if c.name == "boundedness"]
    assert boundedness and not boundedness[0].passed


def test_the_expansion_agrees_with_a_numerical_solution_to_the_claimed_order() -> None:
    order = 2
    r = perturb(
        *GROWTH["equations"], order=order, **{k: v for k, v in GROWTH.items() if k != "equations"}
    )
    assert r.status is Status.OK
    t, eps = sp.Symbol("t", real=True), sp.Symbol("epsilon")
    poly = expansion(r)
    errors = []
    for e in (0.1, 0.05):
        numeric = solve_numerical(
            OdeNumericInput(
                equations=[r"y' = -y + e y^2".replace(" e ", " \\epsilon ")],
                independent="t",
                conditions=["y(0) = 1"],
                parameters={"epsilon": e},
                t_end=1.5,
                rtol=1e-12,
                atol=1e-14,
            ),
            L,
        )
        assert numeric.status is Status.OK, numeric.message
        errors.append(abs(float(poly.subs({eps: e, t: 1.5})) - numeric.series[0].values[-1]))
    assert errors[0] / errors[1] > 2 ** (order + 1) * 0.6


def test_poincare_lindstedt_gives_the_known_duffing_frequency_series() -> None:
    r = perturb(
        r"y'' + y + \epsilon y^3 = 0",
        independent="t",
        small_parameter="epsilon",
        equilibrium="0",
    )
    assert r.status is Status.OK, r.message
    (item,) = r.solutions
    assert item.expansion is not None and item.expansion.kind == "poincare_lindstedt"
    frequency = item.equations_latex[1]
    assert r"1 + \frac{3 \epsilon A^{2}}{8} - \frac{21 \epsilon^{2} A^{4}}{256}" in frequency
    assert item.verification.passed
    assert {c.name for c in item.expansion.checks} == {
        "residual_order",
        "initial_conditions",
        "boundedness",
    }
    assert all(
        c.passed for c in item.expansion.checks
    )  # no secular term, unlike the regular expansion
    assert any("displacement from the equilibrium" in n for n in r.notes)


# --- the checker rejects wrong expansions ------------------------------------------------------


def _check(candidate_terms: list[sp.Expr], order: int) -> bool:
    problem = read_ode(OdeSolveInput(**GROWTH), L)
    eps = next(p for p in problem.parameters if p.name == "epsilon")
    candidate = sum((eps**k * term for k, term in enumerate(candidate_terms)), sp.Integer(0))
    verification, _, conditions = _check_expansion(
        problem, candidate, eps, order, [("y", 0, sp.Integer(0), sp.Integer(1))]
    )
    return verification.passed and all(c.satisfied for c in conditions)


def test_a_wrong_order_of_the_expansion_is_caught() -> None:
    t = sp.Symbol("t", real=True)
    y0 = sp.exp(-t)
    y1 = (1 - sp.exp(-t)) * sp.exp(-t)
    y2 = (1 - 2 * sp.exp(-t) + sp.exp(-2 * t)) * sp.exp(-t)
    assert _check([y0, y1, y2], 2)
    assert not _check([y0, y1 * (1 + sp.Rational(1, 100)), y2], 2)
    assert not _check([y0, y1, y2 * (1 + sp.Rational(1, 100))], 2)
    assert not _check([y0, -y1, y2], 2)
    assert not _check([y0, y1], 2)  # claims one order more than it has
    assert not _check([y0 * 2, y1, y2], 2)  # violates y(0) = 1 and the equation


# --- no small parameter, no expansion ---------------------------------------------------------


def test_the_small_parameter_is_never_chosen_for_the_user() -> None:
    r = perturb(
        r"y'' + y + \epsilon y^3 = 0", independent="t", conditions=["y(0) = 1", "y'(0) = 0"]
    )
    assert r.status is Status.PARSE_ERROR and "Name the small parameter" in r.message
    assert "epsilon" in r.message  # the candidates are listed, none is picked
    assert r.solutions == []
    r = perturb(
        r"y'' + y + \epsilon y^3 = 0",
        independent="t",
        conditions=["y(0) = 1", "y'(0) = 0"],
        small_parameter="zeta",
    )
    assert r.status is Status.PARSE_ERROR and "not a parameter" in r.message


@pytest.mark.parametrize(
    ("kwargs", "status", "fragment"),
    [
        (
            {"equations": [r"y' = -y + \epsilon y^2"], "conditions": [], "order": 2},
            Status.PARSE_ERROR,
            "every initial value",
        ),
        (
            {
                "equations": [r"y' = y^2 + t + \epsilon y"],
                "conditions": ["y(0) = 1"],
                "order": 2,
            },
            Status.UNSUPPORTED,
            "no closed form",
        ),
        (
            {"equations": [r"y' = -y + \epsilon y^2"], "conditions": ["y(0) = 1"], "order": 5},
            Status.UNSUPPORTED,
            "implemented to order 4",
        ),
        (
            {
                "equations": [r"\dot{x} = -y + \epsilon x", r"\dot{y} = x"],
                "conditions": ["x(0) = 1", "y(0) = 0"],
                "order": 2,
            },
            Status.UNSUPPORTED,
            "one equation",
        ),
    ],
)
def test_regular_expansions_refuse_what_they_cannot_do(
    kwargs: dict[str, Any], status: Status, fragment: str
) -> None:
    order = kwargs.pop("order")
    r = perturb(
        *kwargs.pop("equations"), order=order, independent="t", small_parameter="epsilon", **kwargs
    )
    assert r.status is status and fragment in r.message
    assert r.solutions == []


@pytest.mark.parametrize(
    ("equation", "equilibrium", "fragment"),
    [
        (r"y'' + 0.1 y' + y + \epsilon y^3 = 0", "0", "depends on the velocity"),
        (r"y'' + y + \epsilon y^3 = \cos t", "0", "depends on the velocity or explicitly"),
        (r"y' + \epsilon y = 0", "0", "second-order"),
        (r"y'' + \sin(\sin(y)) + \epsilon y = 0", "0", "potential"),
        (r"y'' + y + 1 + \epsilon y^3 = 0", "0", "equilibrium"),
    ],
)
def test_periodic_expansions_refuse_damping_forcing_and_a_wrong_equilibrium(
    equation: str, equilibrium: str, fragment: str
) -> None:
    r = perturb(equation, independent="t", small_parameter="epsilon", equilibrium=equilibrium)
    assert r.status in (Status.UNSUPPORTED, Status.PARSE_ERROR), r.message
    assert fragment in r.message
    assert r.solutions == []


def test_the_engine_dispatches_perturbation(engine: Engine) -> None:
    r = engine.ode_solve_symbolic(
        OdeSolveInput(
            equations=[r"y'' + y + \epsilon y^3 = 0"],
            independent="t",
            method="perturbation",
            small_parameter="epsilon",
            equilibrium="0",
        )
    )
    assert r.status is Status.OK and r.method == "perturbation"
    assert r.solutions[0].expansion is not None
    assert r.solutions[0].expansion.kind == "poincare_lindstedt"
