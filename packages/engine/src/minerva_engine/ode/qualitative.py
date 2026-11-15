"""Equilibria, linearisation and local stability of an autonomous system, and its direction field.

Everything here is a statement about the **linearisation at an equilibrium**, which is local: an
eigenvalue with a negative real part says that orbits starting close enough to that point approach it
(for a hyperbolic point), and says nothing about orbits that start elsewhere or about the nonlinear
system's global behaviour. Where an eigenvalue has zero real part the linearisation does not decide,
and where the signs depend on parameters that were not given the classification is *undecided*, not
guessed.

* Equilibria come from solving f(x) = 0 exactly; each one is confirmed by substituting it back. The list
  is "equilibria found": complete only when f is polynomial (then the solver is exhaustive), and a
  continuum of equilibria is not listed.
* The Jacobian comes from differentiating f symbolically and is confirmed against central differences
  of f taken numerically at random points (mpmath), which share nothing with the symbolic route.
* Eigenvalues are exact for one or two states; with numeric parameters they are also computed by an
  eigenvalue routine (NumPy) and checked against the characteristic polynomial.
* The direction field is f evaluated on a grid in a window the user chose; it is a picture of f, not of
  any solution.
"""

from __future__ import annotations

import random

import mpmath as mp
import numpy as np
import sympy as sp

from ..latexprint import to_latex
from ..models import Status
from .analyze import is_zero
from .models import (
    DirectionField,
    Equilibrium,
    OdeQualitative,
)
from .numerical import parameter_problem
from .reading import OdeProblem, _display
from .symbolic import SolverTimeout, deadline
from .verification import DIGITS, Unevaluable, _lambdify, _parameter_values

MAX_STATES = 3
MAX_EQUILIBRIA = 12
GRID = 15
ZERO_REAL_PART = 1e-9

LOCAL = (
    "Local and from the linearisation only: it describes orbits that start close to the point and says "
    "nothing about global behaviour or about orbits that start elsewhere."
)

_CLASSES = {
    "stable_node": "linearly asymptotically stable (a sink: a node)",
    "stable_spiral": "linearly asymptotically stable (a sink: a spiral)",
    "unstable_node": "linearly unstable (a source: a node)",
    "unstable_spiral": "linearly unstable (a source: a spiral)",
    "saddle": "linearly unstable (a saddle: some directions attract, some repel)",
    "centre": "eigenvalues on the imaginary axis: the linearisation does not decide (a centre of the "
    "linear system; the nonlinear terms decide whether orbits close up or spiral)",
    "non_hyperbolic": "an eigenvalue has zero real part: the linearisation does not decide",
    "undecided": "the signs of the eigenvalues depend on parameters that were not given: undecided",
}


def first_order_field(
    problem: OdeProblem, rhs: list[sp.Expr]
) -> tuple[list[tuple[str, int]], list[sp.Symbol], list[sp.Expr]]:
    """The equivalent first-order system x' = f(x): states (u, k) for k below the order, and f."""
    labels = [(u, k) for u in problem.unknowns for k in range(problem.orders[u])]
    states = [problem.jets[label] for label in labels]
    field: list[sp.Expr] = []
    for u in problem.unknowns:
        n = problem.orders[u]
        for k in range(n):
            field.append(problem.jets[(u, k + 1)] if k < n - 1 else rhs[problem.unknowns.index(u)])
    return labels, states, field


def _jacobian_check(
    problem: OdeProblem,
    states: list[sp.Symbol],
    field: list[sp.Expr],
    jacobian: sp.Matrix,
    seed: int = 20260501,
) -> bool:
    """The symbolic Jacobian against central differences of f taken numerically at random points."""
    mp.mp.dps = DIGITS
    args = [*states, *problem.parameters]
    try:
        f = _lambdify(args, sp.Tuple(*field))
        j = _lambdify(args, jacobian)
    except Unevaluable:
        return False
    rng = random.Random(seed)  # noqa: S311 - deterministic sampling, not cryptographic
    good = 0
    for _ in range(12):
        point = [mp.mpf(rng.uniform(-2.0, 2.0)) for _ in states]
        parameters = _parameter_values(problem, rng, wide=False)
        h = mp.mpf(10) ** -18
        try:
            analytic = mp.matrix(j(*point, *parameters))
            worst = mp.mpf(0)
            scale = mp.mpf(1)
            for column in range(len(states)):
                up = list(point)
                down = list(point)
                up[column] += h
                down[column] -= h
                numeric = [
                    (a - b) / (2 * h)
                    for a, b in zip(f(*up, *parameters), f(*down, *parameters), strict=True)
                ]
                for row in range(len(states)):
                    scale = max(scale, abs(analytic[row, column]))
                    worst = max(worst, abs(analytic[row, column] - numeric[row]))
        except (ArithmeticError, ValueError, TypeError, ZeroDivisionError):
            continue
        if worst > mp.mpf(10) ** -9 * scale:
            return False
        good += 1
    return good >= 4


def _equilibria(
    states: list[sp.Symbol], field: list[sp.Expr], numeric: bool, timeout: float
) -> tuple[list[dict[sp.Symbol, sp.Expr]], bool, list[str]]:
    """Isolated equilibria found by solving f = 0, each confirmed by substitution; whether the list is
    complete; notes."""
    notes: list[str] = []
    polynomial = all(f.is_polynomial(*states) for f in field)
    try:
        with deadline(timeout):
            found = sp.solve(field, states, dict=True)
    except SolverTimeout:
        return [], False, ["Solving for the equilibria did not finish within the time limit."]
    except (NotImplementedError, ValueError, TypeError, ArithmeticError, RecursionError):
        return [], False, ["SymPy found no closed form for the equilibria."]
    confirmed: list[dict[sp.Symbol, sp.Expr]] = []
    continuum = False
    for solution in found:
        if set(solution) != set(states) or any(v.has(*states) for v in solution.values()):
            continuum = True
            continue
        if numeric and any(abs(sp.im(sp.N(v, 30))) > 1e-12 for v in solution.values()):
            continue
        if all(is_zero(f.xreplace(solution)) for f in field):
            confirmed.append(solution)
    if continuum:
        notes.append("A continuum of equilibria (not isolated points) was found and is not listed.")
    if not numeric and any(any(v.has(sp.I) for v in s.values()) for s in confirmed):
        notes.append(
            "Some solutions are complex for real parameters; only real ones are equilibria."
        )
    complete = polynomial and not continuum
    if not polynomial:
        notes.append(
            "The field is not polynomial: these are the equilibria found; there may be others."
        )
    return confirmed[:MAX_EQUILIBRIA], complete, notes


def _classify(eigenvalues: list[complex]) -> str:
    scale = max(1.0, *(abs(e) for e in eigenvalues))
    real = [e.real for e in eigenvalues]
    if any(abs(r) <= ZERO_REAL_PART * scale for r in real):
        return (
            "centre"
            if all(abs(r) <= ZERO_REAL_PART * scale for r in real)
            and any(abs(e.imag) > ZERO_REAL_PART * scale for e in eigenvalues)
            else "non_hyperbolic"
        )
    complex_pair = any(abs(e.imag) > ZERO_REAL_PART * scale for e in eigenvalues)
    if all(r < 0 for r in real):
        return "stable_spiral" if complex_pair else "stable_node"
    if all(r > 0 for r in real):
        return "unstable_spiral" if complex_pair else "unstable_node"
    return "saddle"


def _symbolic_class(eigenvalues: list[sp.Expr]) -> str:
    """Decide from the signs SymPy can prove under the declared assumptions, else undecided."""
    real = [sp.re(e) for e in eigenvalues]
    complex_pair = any(sp.im(e).is_zero is not True for e in eigenvalues)
    if all(r.is_zero is True for r in real):
        return "centre" if complex_pair else "non_hyperbolic"
    if any(r.is_zero is True for r in real):
        return "non_hyperbolic"
    if all(r.is_negative is True for r in real):
        return "stable_spiral" if complex_pair else "stable_node"
    if all(r.is_positive is True for r in real):
        return "unstable_spiral" if complex_pair else "unstable_node"
    if any(r.is_negative is True for r in real) and any(r.is_positive is True for r in real):
        return "saddle"
    return "undecided"


def _numeric_eigenvalues(matrix: sp.Matrix) -> list[complex] | None:
    try:
        values = np.linalg.eigvals(np.array(matrix.evalf(30).tolist(), dtype=complex))
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return None
    out = [complex(v) for v in values]
    # confirmed against the characteristic polynomial, which the routine did not use
    lam = sp.Symbol("lam")
    poly = (matrix - lam * sp.eye(matrix.shape[0])).det()
    scale = max(1.0, *(abs(v) for v in out)) ** matrix.shape[0]
    for v in out:
        if abs(complex(sp.N(poly.subs(lam, v), 20))) > 1e-8 * scale:
            return None
    return sorted(out, key=lambda z: (z.real, z.imag))


def _number(value: complex) -> str:
    if abs(value.imag) < 1e-12 * max(1.0, abs(value)):
        return f"{value.real:.8g}"
    sign = "+" if value.imag >= 0 else "-"
    return f"{value.real:.8g} {sign} {abs(value.imag):.8g}i"


def _direction_field(
    states: list[sp.Symbol], field: list[sp.Expr], window: tuple[float, float, float, float]
) -> DirectionField | None:
    x0, x1, y0, y1 = window
    try:
        from .verification import lambdify_numeric

        fn = lambdify_numeric(states, sp.Tuple(*field), "numpy")
    except Unevaluable:
        return None
    xs = np.linspace(x0, x1, GRID)
    ys = np.linspace(y0, y1, GRID)
    grid_x, grid_y = np.meshgrid(xs, ys)
    with np.errstate(all="ignore"):
        try:
            out = fn(grid_x.ravel(), grid_y.ravel())
            dx = np.broadcast_to(np.asarray(out[0], dtype=float), grid_x.ravel().shape)
            dy = np.broadcast_to(np.asarray(out[1], dtype=float), grid_x.ravel().shape)
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            return None
    keep = np.isfinite(dx) & np.isfinite(dy)
    speed = np.hypot(dx, dy)

    def clean(values: np.ndarray) -> list[float]:
        return [float(f"{v:.8g}") for v in values[keep]]

    return DirectionField(
        window=[x0, x1, y0, y1],
        x=clean(grid_x.ravel()),
        y=clean(grid_y.ravel()),
        dx=clean(dx),
        dy=clean(dy),
        speed=clean(speed),
    )


def qualitative(
    problem: OdeProblem,
    rhs: list[sp.Expr],
    parameters: dict[str, float],
    window: tuple[float, float, float, float] | None,
    timeout: float,
) -> OdeQualitative:
    """The equilibria of the autonomous system, their linearisations and local stability."""
    labels, states, field = first_order_field(problem, rhs)
    if len(states) > MAX_STATES:
        return OdeQualitative(
            status=Status.UNSUPPORTED, message=f"At most {MAX_STATES} states are supported."
        )
    names = [_display(u, k) for u, k in labels]
    numeric = (
        bool(parameters) or not problem.parameters
    )  # every parameter has a value (or there is none)
    if problem.parameters and parameters:
        issue = parameter_problem(problem, parameters)
        if issue:
            return OdeQualitative(status=Status.PARSE_ERROR, message=issue)
        # exact decimals (0.2 is 1/5): equilibria of a polynomial system are found exactly
        substitution = {p: sp.Rational(repr(parameters[p.name])) for p in problem.parameters}
        field = [f.xreplace(substitution) for f in field]
    elif problem.parameters:
        numeric = False
    state_latex = [to_latex(s, symbol_names=problem.display) for s in states]
    jacobian = sp.Matrix(field).jacobian(states)
    if not _jacobian_check(problem, states, field, jacobian):
        return OdeQualitative(
            status=Status.NOT_VERIFIED,
            message="The Jacobian did not agree with numerical differences of the field, so nothing is shown.",
        )
    solutions, complete, notes = _equilibria(
        states, field, (numeric and bool(parameters)) or numeric, timeout
    )
    equilibria: list[Equilibrium] = []
    for solution in solutions:
        point = jacobian.xreplace(solution)
        eigenvalues_symbolic: list[sp.Expr] = []
        if point.shape[0] <= 2:
            try:
                eigenvalues_symbolic = [
                    sp.simplify(e)
                    for e, mult in point.eigenvals().items()
                    for _ in range(int(mult))
                ]
            except (NotImplementedError, ValueError, TypeError):
                eigenvalues_symbolic = []
        numeric_values: list[complex] | None = None
        if not point.free_symbols:
            numeric_values = _numeric_eigenvalues(point)
        if numeric_values:
            kind = _classify(numeric_values)
        elif eigenvalues_symbolic:
            kind = _symbolic_class(eigenvalues_symbolic)
        else:
            kind = "undecided"
        lam = sp.Symbol("lambda")
        characteristic = (lam * sp.eye(point.shape[0]) - point).det()
        located = [solution[state] for state in states]
        equilibria.append(
            Equilibrium(
                point_numeric=(
                    [float(sp.N(v, 15)) for v in located]
                    if not any(v.free_symbols for v in located)
                    else []
                ),
                point_latex=[
                    f"{name} = {to_latex(solution[state], symbol_names=problem.display)}"
                    for name, state in zip(names, states, strict=True)
                ],
                jacobian_latex=to_latex(point),
                eigenvalues_latex=[to_latex(e) for e in eigenvalues_symbolic],
                eigenvalues_numeric=[_number(v) for v in numeric_values] if numeric_values else [],
                characteristic_latex=to_latex(
                    sp.expand(characteristic), symbol_names={lam: "\\lambda"}
                ),
                classification=kind,
                classification_text=_CLASSES[kind],
                decided=kind != "undecided",
            )
        )
    field_direction = None
    if window is not None and len(states) == 2 and (numeric or not problem.parameters):
        field_direction = _direction_field(states, field, window)
    if window is not None and len(states) != 2:
        notes.append("A phase plane needs exactly two states.")
    if window is not None and problem.parameters and not parameters:
        notes.append("The direction field needs a value for every parameter.")
    if problem.parameters and not parameters:
        notes.append(
            "Parameters have no values: eigenvalues are exact, and the stability may be undecided."
        )
    return OdeQualitative(
        status=Status.OK,
        states_latex=state_latex,
        field_latex=[
            f"{name}' = {to_latex(f, symbol_names=problem.display)}"
            for name, f in zip(names, field, strict=True)
        ],
        equilibria=equilibria,
        complete=complete,
        jacobian_checked=True,
        direction_field=field_direction,
        notes=notes,
        local_note=LOCAL,
    )
