"""Reference problems for the ODE framework: answers that do not come from Minerva.

Each problem is judged by a route that shares no code with the solver it checks: a closed form written
down by hand, Taylor's method in mpmath (`mp.odefun`), the matrix exponential (`mp.expm`), the Airy
functions, a quarter-period integral, published values (Robertson's problem, Lorenz's equilibria), a
conserved quantity computed from the returned trajectory, or arithmetic. The corpus is part of
`reference_benchmark.PROBLEMS`; `python -m minerva_engine.reference_benchmark --check` fails on a wrong
answer, on an answer that was lost, and on an answer where a refusal was due.

Refusals are problems too: a result for a problem that has no result (a blow-up inside the interval, a
series about a singular point, a delay, numbers the user did not give) would be wrong or invented.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import mpmath as mp
import sympy as sp

from ..models import Limits
from ..reference_core import Problem, _verdict
from ..service import Engine
from ..textparse import parse_text
from .models import OdeAnalyzeInput, OdeInput, OdeNumericInput, OdeSolveInput

Judge = Callable[[Engine], tuple[str, str]]


def _values(text: str, symbol: str, points: list[float], **params: float) -> list[complex]:
    """The returned right-hand side (text syntax) evaluated by mpmath at `points`."""
    expression = parse_text(text, {symbol: ["real"]}, Limits())
    names = sorted(expression.free_symbols, key=lambda s: s.name)
    fn = sp.lambdify(names, expression, modules="mpmath")
    out = []
    for x in points:
        args = [mp.mpf(x) if s.name == symbol else mp.mpf(params[s.name]) for s in names]
        out.append(complex(fn(*args)))
    return out


def _agree(got: list[complex], want: list[Any], tol: float = 1e-9) -> bool:
    return all(
        abs(g - complex(w)) <= tol * max(1.0, abs(complex(w)))
        for g, w in zip(got, want, strict=True)
    )


def _closed_form(
    engine: Engine,
    equations: list[str],
    conditions: list[str],
    variable: str,
    exact: Callable[[float], Any],
    points: list[float],
    **extra: Any,
) -> tuple[str, str]:
    r = engine.ode_solve_symbolic(
        OdeSolveInput(equations=equations, independent=variable, conditions=conditions, **extra)
    )
    if r.status.value != "ok" or not r.solutions:
        return _verdict("correct", False, None, f"status {r.status.value}")
    # A branch of the solution may be the one for these conditions: one of those shown must agree.
    agrees = any(
        _agree(_values(item.rhs_text[0], variable, points), [exact(x) for x in points])
        for item in r.solutions
        if item.rhs_text
    )
    return _verdict("correct", True, agrees, "closed form vs the known solution")


def _numeric(
    engine: Engine,
    equations: list[str],
    conditions: list[str],
    t_end: float,
    exact: Callable[[float], list[Any]],
    parameters: dict[str, float] | None = None,
    variable: str = "t",
    tol: float = 1e-6,
    **extra: Any,
) -> tuple[str, str]:
    r = engine.ode_solve_numerical(
        OdeNumericInput(
            equations=equations,
            independent=variable,
            conditions=conditions,
            t_end=t_end,
            parameters=parameters or {},
            **extra,
        )
    )
    if r.status.value != "ok" or not r.series:
        return _verdict("correct", False, None, f"status {r.status.value}")
    agrees = True
    for index in (25, 100, len(r.t) - 1):
        want = exact(r.t[index])
        got = [r.series[k].values[index] for k in range(len(want))]
        agrees = agrees and all(
            abs(g - float(w)) <= tol * max(1.0, abs(float(w)))
            for g, w in zip(got, want, strict=True)
        )
    return _verdict("correct", True, agrees, "trajectory vs the reference")


# --- first-order and linear equations: closed forms ---------------------------------------------


def _separable(e: Engine) -> tuple[str, str]:
    return _closed_form(
        e, ["y' = -2 x y^2"], ["y(0) = 1"], "x", lambda x: 1 / (1 + x**2), [0.3, 1.1, 2.0]
    )


def _bernoulli(e: Engine) -> tuple[str, str]:
    return _closed_form(
        e, ["y' + y = y^2"], ["y(0) = 1/2"], "x", lambda x: 1 / (1 + mp.exp(x)), [0.2, 0.9, 1.7]
    )


def _linear2(e: Engine) -> tuple[str, str]:
    return _closed_form(
        e,
        ["y'' + 3 y' + 2 y = 0"],
        ["y(0) = 1", "y'(0) = 0"],
        "t",
        lambda t: 2 * mp.exp(-t) - mp.exp(-2 * t),
        [0.3, 1.1, 2.4],
    )


def _euler_cauchy(e: Engine) -> tuple[str, str]:
    return _closed_form(
        e,
        ["x^2 y'' + x y' - y = 0"],
        ["y(1) = 1", "y'(1) = 0"],
        "x",
        lambda x: (x + 1 / x) / 2,
        [0.5, 1.5, 3.0],
    )


# --- oscillators and nonlinear initial-value problems -------------------------------------------


def _harmonic(e: Engine) -> tuple[str, str]:
    return _numeric(
        e,
        [r"y'' + \omega^2 y = 0"],
        ["y(0) = 1", "y'(0) = 0"],
        10,
        lambda t: [mp.cos(2 * t), -2 * mp.sin(2 * t)],
        {"omega": 2.0},
    )


def _damped(e: Engine) -> tuple[str, str]:
    # y'' + 2 z w y' + w^2 y = 0, w = 3, z = 0.1, y(0) = 1, y'(0) = 0 (underdamped, exact solution)
    w, z = 3.0, 0.1
    wd = w * (1 - z * z) ** 0.5

    def exact(t: float) -> list[Any]:
        decay = mp.exp(-z * w * t)
        y = decay * (mp.cos(wd * t) + z * w / wd * mp.sin(wd * t))
        return [y]

    return _numeric(
        e,
        [r"y'' + 2 z w y' + w^2 y = 0"],
        ["y(0) = 1", "y'(0) = 0"],
        8,
        exact,
        {"z": z, "w": w},
        tol=1e-6,
    )


def _pendulum(e: Engine) -> tuple[str, str]:
    reference = mp.odefun(lambda t, s: [s[1], -mp.sin(s[0])], 0, [mp.mpf("1.5"), 0])
    return _numeric(
        e,
        [r"\theta'' + \sin\theta = 0"],
        [r"\theta(0) = 1.5", r"\theta'(0) = 0"],
        8,
        lambda t: [reference(t)[0]],
        rtol=1e-10,
        atol=1e-12,
    )


def _logistic(e: Engine) -> tuple[str, str]:
    return _numeric(
        e,
        [r"y' = y \cdot (1 - y)"],
        ["y(0) = 1/10"],
        12,
        lambda t: [1 / (1 + 9 * mp.exp(-t))],
    )


# --- stiff and coupled systems ------------------------------------------------------------------


def _robertson(e: Engine) -> tuple[str, str]:
    """Robertson's chemical kinetics problem: the values at t = 40 are the standard published ones."""
    r = e.ode_solve_numerical(
        OdeNumericInput(
            equations=[
                r"\dot{a} = -0.04 a + 10^4 b c",
                r"\dot{b} = 0.04 a - 10^4 b c - 3 \cdot 10^7 b^2",
                r"\dot{c} = 3 \cdot 10^7 b^2",
            ],
            conditions=["a(0) = 1", "b(0) = 0", "c(0) = 0"],
            t_end=40,
        )
    )
    if r.status.value != "ok":
        return _verdict("correct", False, None, f"status {r.status.value}")
    a, b, c = (s.values[-1] for s in r.series)
    agrees = (
        abs(a - 0.7158270687) < 1e-6
        and abs(b - 9.185534764e-6) < 1e-9
        and abs(c - 0.2841637457) < 1e-6
    )
    return _verdict("correct", True, agrees, "published values at t = 40")


def _stiff_linear(e: Engine) -> tuple[str, str]:
    return _numeric(
        e,
        [r"y' = -1000 \cdot (y - \cos t) - \sin t"],
        ["y(0) = 1"],
        5,
        lambda t: [mp.cos(t)],
    )


def _coupled_linear(e: Engine) -> tuple[str, str]:
    matrix = mp.matrix([[1, 2], [-3, -1]])
    start = mp.matrix([1, 2])
    r = e.ode_solve_numerical(
        OdeNumericInput(
            equations=["x' = x + 2*y", "y' = -3*x - y"],
            format="text",
            independent="t",
            conditions=["x(0) = 1", "y(0) = 2"],
            t_end=2,
            rtol=1e-10,
            atol=1e-12,
        )
    )
    if r.status.value != "ok":
        return _verdict("correct", False, None, f"status {r.status.value}")
    final = mp.expm(matrix * mp.mpf(2)) * start
    agrees = (
        abs(r.series[0].values[-1] - float(final[0])) < 1e-7
        and abs(r.series[1].values[-1] - float(final[1])) < 1e-7
    )
    return _verdict("correct", True, agrees, "trajectory vs the matrix exponential")


def _named_system(e: Engine) -> tuple[str, str]:
    """A system written the way a model writes one: `dy1/dt`, names `y1`, `y2`, text format."""
    r = e.ode_solve_numerical(
        OdeNumericInput(
            equations=["dy1/dt = y2", "dy2/dt = -y1"],
            format="text",
            independent="t",
            unknowns=["y1", "y2"],
            conditions=[
                {"unknown": "y1", "at": 0, "value": 1},  # type: ignore[list-item]
                {"unknown": "y2", "at": 0, "value": 0},  # type: ignore[list-item]
            ],
            t_end=6,
            rtol=1e-10,
            atol=1e-12,
        )
    )
    if r.status.value != "ok" or len(r.series) != 2:
        return _verdict("correct", False, None, f"status {r.status.value}")
    # cos t and -sin t solve it with y1(0) = 1, y2(0) = 0 (plain arithmetic, no ODE solver)
    last = r.t[-1]
    agrees = (
        abs(r.series[0].values[-1] - mp.cos(last)) < 1e-7
        and abs(r.series[1].values[-1] + mp.sin(last)) < 1e-7
    )
    return _verdict("correct", True, bool(agrees), "trajectory vs cos t and -sin t")


def _lotka_volterra(e: Engine) -> tuple[str, str]:
    """V = d x - c ln x + b y - a ln y is constant along solutions of the Lotka-Volterra equations."""
    a, b, c, d = 1.0, 0.5, 0.75, 0.25
    r = e.ode_solve_numerical(
        OdeNumericInput(
            equations=[r"\dot{x} = a x - b x y", r"\dot{y} = -c y + d x y"],
            conditions=["x(0) = 2", "y(0) = 1"],
            parameters={"a": a, "b": b, "c": c, "d": d},
            t_end=30,
            rtol=1e-10,
            atol=1e-12,
        )
    )
    if r.status.value != "ok":
        return _verdict("correct", False, None, f"status {r.status.value}")
    v = [
        d * x - c * mp.log(x) + b * y - a * mp.log(y)
        for x, y in zip(r.series[0].values, r.series[1].values, strict=True)
    ]
    return _verdict(
        "correct", True, max(abs(x - v[0]) for x in v) < 1e-7, "a conserved quantity of the flow"
    )


# --- boundary-value problems --------------------------------------------------------------------


def _bvp(
    e: Engine, equations: list[str], conditions: list[str], exact: Callable[[float], Any], **kw: Any
) -> tuple[str, str]:
    r = e.ode_solve_numerical(
        OdeNumericInput(
            equations=equations, independent="x", kind="bvp", conditions=conditions, **kw
        )
    )
    if r.status.value != "ok" or not r.series:
        return _verdict("correct", False, None, f"status {r.status.value}")
    agrees = all(
        abs(r.series[0].values[i] - float(exact(r.t[i]))) < 1e-6 for i in (0, 50, 100, len(r.t) - 1)
    )
    return _verdict("correct", True, agrees, "solution vs the closed form")


def _bvp_linear(e: Engine) -> tuple[str, str]:
    return _bvp(e, ["y'' = -y"], ["y(0) = 0", "y(1) = 1"], lambda x: mp.sin(x) / mp.sin(1))


def _bvp_airy(e: Engine) -> tuple[str, str]:
    system = mp.matrix([[mp.airyai(0), mp.airybi(0)], [mp.airyai(2), mp.airybi(2)]])
    a, b = mp.lu_solve(system, mp.matrix([1, mp.mpf("0.1")]))
    return _bvp(
        e,
        ["y'' = x*y"],
        ["y(0) = 1", "y(2) = 0.1"],
        lambda x: a * mp.airyai(x) + b * mp.airybi(x),
        format="text",
    )


def _bvp_without_a_solution(e: Engine) -> tuple[str, str]:
    r = e.ode_solve_numerical(
        OdeNumericInput(
            equations=["y'' = -y"],
            independent="x",
            kind="bvp",
            conditions=["y(0) = 0", "y(3.14159265358979) = 1"],
        )
    )
    return _verdict("refused", r.status.value == "ok", None, f"status {r.status.value}")


# --- series, perturbation, qualitative -----------------------------------------------------------


def _series(e: Engine) -> tuple[str, str]:
    r = e.ode_solve_symbolic(
        OdeSolveInput(
            equations=["y' = y^2"],
            independent="x",
            conditions=["y(0) = 1"],
            method="series",
            order=8,
        )
    )
    if r.status.value != "ok":
        return _verdict("correct", False, None, f"status {r.status.value}")
    polynomial = sp.sympify(r.solutions[0].rhs_text[0], locals={"x": sp.Symbol("x")})
    coefficients = [float(polynomial.coeff(sp.Symbol("x"), k)) for k in range(9)]
    return _verdict(
        "correct",
        True,
        coefficients == [float(c) for c in mp.taylor(lambda x: 1 / (1 - x), 0, 8)],
        "Taylor coefficients of 1/(1-x)",
    )


def _series_singular(e: Engine) -> tuple[str, str]:
    r = e.ode_solve_symbolic(
        OdeSolveInput(
            equations=["y' = 1/x"], independent="x", conditions=["y(0) = 1"], method="series"
        )
    )
    return _verdict("refused", r.status.value == "ok", None, f"status {r.status.value}")


def _duffing_frequency(e: Engine) -> tuple[str, str]:
    """The Poincare-Lindstedt frequency against the exact period (quarter-period integral)."""
    r = e.ode_solve_symbolic(
        OdeSolveInput(
            equations=[r"y'' + y + \epsilon y^3 = 0"],
            independent="t",
            method="perturbation",
            small_parameter="epsilon",
            equilibrium="0",
        )
    )
    if r.status.value != "ok":
        return _verdict("correct", False, None, f"status {r.status.value}")
    from ..latexparse import parse_latex

    frequency = parse_latex(r.solutions[0].equations_latex[1].split("=", 1)[1])
    eps = mp.mpf("0.01")
    quarter = mp.quad(lambda s: 1 / mp.sqrt(1 - s**2 + eps * (1 - s**4) / 2), [0, 1])
    exact = 2 * mp.pi / (4 * quarter)
    value = frequency.subs({sp.Symbol("epsilon"): sp.Rational(1, 100), sp.Symbol("A"): 1})
    return _verdict(
        "correct",
        True,
        abs(complex(value.evalf(20)) - complex(exact)) < 1e-5,
        "frequency vs the exact period",
    )


def _lorenz(e: Engine) -> tuple[str, str]:
    r = e.ode_analyze(
        OdeAnalyzeInput(
            equations=["x' = s*(y-x)", "y' = x*(r-z)-y", "z' = x*y-b*z"],
            format="text",
            independent="t",
            detail="qualitative",
            parameters={"s": 10.0, "r": 28.0, "b": 8 / 3},
        )
    )
    q = r.qualitative
    if q is None or len(q.equilibria) != 3:
        return _verdict("correct", False, None, "equilibria not found")
    others = [x for x in q.equilibria if x.point_latex[0] != "x = 0"]
    ok = len(others) == 2
    for eq in others:
        real = sorted(
            complex(v.replace(" ", "").replace("i", "j")).real for v in eq.eigenvalues_numeric
        )
        ok = (
            ok
            and abs(real[0] + 13.8546) < 1e-3
            and abs(real[1] - 0.09396) < 1e-3
            and abs(real[2] - 0.09396) < 1e-3
        )
    return _verdict("correct", True, ok, "eigenvalues of the Lorenz equilibria (published)")


def _pendulum_centre(e: Engine) -> tuple[str, str]:
    r = e.ode_analyze(
        OdeAnalyzeInput(
            equations=[r"\theta'' + \sin\theta = 0"], independent="t", detail="qualitative"
        )
    )
    q = r.qualitative
    if q is None:
        return _verdict("correct", False, None, "no analysis")
    kinds = {x.point_latex[0]: x.classification for x in q.equilibria}
    # eigenvalues +-i at the bottom: the linearisation cannot call it stable; +-1 at the top: a saddle
    return _verdict(
        "correct",
        True,
        kinds == {"\\theta = 0": "centre", "\\theta = \\pi": "saddle"},
        "classification from the eigenvalues",
    )


# --- refusals ---------------------------------------------------------------------------------------


def _blow_up(e: Engine) -> tuple[str, str]:
    """y' = y^2, y(0) = 1 has the solution 1/(1 - t), which is infinite at t = 1: no trajectory to t = 2."""
    r = e.ode_solve_numerical(
        OdeNumericInput(equations=["y' = y^2"], independent="t", conditions=["y(0) = 1"], t_end=2)
    )
    return _verdict("refused", r.status.value == "ok", None, f"status {r.status.value}")


def _delay(e: Engine) -> tuple[str, str]:
    r = e.ode_solve_symbolic(OdeInput(equations=[r"y'(t) = y(t - 1)"], independent="t"))
    return _verdict("refused", r.status.value == "ok", None, f"status {r.status.value}")


def _numbers_not_given(e: Engine) -> tuple[str, str]:
    r = e.ode_solve_numerical(
        OdeNumericInput(equations=["y' = a y"], independent="t", conditions=["y(0) = 1"], t_end=1)
    )
    return _verdict("refused", r.status.value == "ok", None, f"status {r.status.value}")


ODE_PROBLEMS: list[Problem] = [
    Problem(
        "ode-separable-closed-form",
        "ode-analytic",
        "closed form: y' = -2xy^2 gives y = 1/(1 + x^2)",
        "correct",
        _separable,
    ),
    Problem(
        "ode-bernoulli-closed-form",
        "ode-analytic",
        "closed form: Bernoulli y' + y = y^2 with y(0) = 1/2 gives 1/(1 + e^x)",
        "correct",
        _bernoulli,
    ),
    Problem(
        "ode-linear-second-order",
        "ode-analytic",
        "closed form: roots -1 and -2, y = 2e^-t - e^-2t",
        "correct",
        _linear2,
    ),
    Problem(
        "ode-euler-cauchy",
        "ode-analytic",
        "closed form: x^2 y'' + x y' - y = 0 has solutions x and 1/x",
        "correct",
        _euler_cauchy,
    ),
    Problem("ode-harmonic-oscillator", "ode-numeric", "closed form: cos(wt)", "correct", _harmonic),
    Problem(
        "ode-damped-oscillator",
        "ode-numeric",
        "closed form of the underdamped oscillator",
        "correct",
        _damped,
    ),
    Problem(
        "ode-pendulum-nonlinear",
        "ode-numeric",
        "mpmath.odefun (Taylor's method) on theta'' = -sin(theta)",
        "correct",
        _pendulum,
    ),
    Problem(
        "ode-logistic", "ode-numeric", "closed form of the logistic equation", "correct", _logistic
    ),
    Problem(
        "ode-robertson-stiff",
        "ode-stiff",
        "Robertson (1966) chemical kinetics: standard values at t = 40 (Hairer and Wanner, Solving ODEs II)",
        "correct",
        _robertson,
    ),
    Problem(
        "ode-stiff-linear",
        "ode-stiff",
        "closed form: y = cos t solves y' = -1000(y - cos t) - sin t",
        "correct",
        _stiff_linear,
    ),
    Problem(
        "ode-coupled-matrix-exponential",
        "ode-coupled",
        "mpmath.expm on the constant matrix of x' = Ax",
        "correct",
        _coupled_linear,
    ),
    Problem(
        "ode-named-first-order-system",
        "ode-coupled",
        "closed form: y1 = cos t, y2 = -sin t (written dy1/dt = y2, dy2/dt = -y1)",
        "correct",
        _named_system,
    ),
    Problem(
        "ode-lotka-volterra-invariant",
        "ode-coupled",
        "Lotka-Volterra: d x - c ln x + b y - a ln y is constant along solutions",
        "correct",
        _lotka_volterra,
    ),
    Problem("ode-bvp-linear", "ode-boundary", "closed form: sin(x)/sin(1)", "correct", _bvp_linear),
    Problem(
        "ode-bvp-airy",
        "ode-boundary",
        "Airy functions a Ai + b Bi fitted to the two end values (mpmath.airyai, airybi)",
        "correct",
        _bvp_airy,
    ),
    Problem(
        "ode-bvp-no-solution-refused",
        "ode-boundary",
        "y'' = -y has only c sin(x): y(0) = 0, y(pi) = 1 has no solution",
        "refused",
        _bvp_without_a_solution,
    ),
    Problem(
        "ode-series-geometric",
        "ode-approximation",
        "Taylor coefficients of 1/(1 - x) (mpmath.taylor)",
        "correct",
        _series,
    ),
    Problem(
        "ode-series-singular-point-refused",
        "ode-approximation",
        "y' = 1/x is not defined at 0: no series about it",
        "refused",
        _series_singular,
    ),
    Problem(
        "ode-perturbation-duffing-frequency",
        "ode-approximation",
        "Landau and Lifshitz, Mechanics, section 28, against the exact period (quarter-period integral)",
        "correct",
        _duffing_frequency,
    ),
    Problem(
        "ode-lorenz-equilibria",
        "ode-qualitative",
        "Lorenz (1963), Strogatz, Nonlinear Dynamics and Chaos, section 9.2: eigenvalues at the standard parameters",
        "correct",
        _lorenz,
    ),
    Problem(
        "ode-pendulum-equilibria",
        "ode-qualitative",
        "eigenvalues +-i (centre of the linearisation) and +-1 (saddle)",
        "correct",
        _pendulum_centre,
    ),
    Problem(
        "ode-blow-up-refused",
        "ode-refusal",
        "y' = y^2, y(0) = 1: y = 1/(1 - t) is infinite at t = 1",
        "refused",
        _blow_up,
    ),
    Problem(
        "ode-delay-refused",
        "ode-refusal",
        "a delay equation is not an ordinary differential equation",
        "refused",
        _delay,
    ),
    Problem(
        "ode-numbers-not-given-refused",
        "ode-refusal",
        "Minerva does not choose a parameter value for the user",
        "refused",
        _numbers_not_given,
    ),
]
