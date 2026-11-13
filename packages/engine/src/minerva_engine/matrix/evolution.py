"""Evolution operators: dU/dt = A(t) U(t), U(t0) = I.

Two methods, chosen by what is established about the generator A(t):

* `evolution_commuting`: A(t1) A(t2) = A(t2) A(t1) for all times (constant generators, scalar
  functions of t times commuting matrices, explicit entries that commute with themselves at all
  times, or a time-dependent matrix declared to commute with itself). Then U(t) = exp(integral of
  A from t0 to t). An abstract time-dependent generator gives a representation (the integral of a
  matrix function is not evaluated), never an evaluated matrix.
* `evolution_numeric`: explicit entries with numeric time dependence and no commutation: the system
  is integrated numerically on a span the user supplies, and shown only if three independent checks
  pass. The result is a measurement on that span, not a closed form.

Not supported (reported, never approximated): time-ordered exponentials, Dyson series and Magnus
expansions for abstract generators that do not commute with themselves at different times.
"""

from __future__ import annotations

import cmath
import itertools
import math
import time
from collections.abc import Callable

import mpmath as mp
import sympy as sp

from ..errors import LimitError
from ..latexprint import to_latex
from .expressions import Exp, Sym, Term, Terms, render_terms
from .methods import (
    Attempt,
    Outcome,
    Problem,
    SymbolicTimeout,
    _entries,
    _numeric_candidate,
    _simplified,
    explicit_value,
    matrices_in,
    symbolic_budget,
)
from .models import MatrixBlocked, MatrixStep
from .verify import (
    Count,
    check_derivative,
    close,
    numeric_points,
    poly_diff,
    poly_equal,
    poly_mul,
    poly_subs,
    series_of_terms,
    telemetry,
)

MAX_SIZE = 4
SAMPLES = 4  # output times after the initial one
RTOL, ATOL = 1e-12, 1e-14
METHOD_TOL = 1e-7
REVERSAL_TOL = 1e-8
LIOUVILLE_TOL = 1e-8
MAX_ATTEMPTS = 400_000
MIN_RK4_STEPS = 2_000
MAX_RK4_STEPS = 60_000
NOT_CLOSED_FORM = (
    "the generator does not commute with itself at different times, so U(t) is a time-ordered "
    "exponential; time-ordered exponentials, Dyson series and Magnus expansions are not supported "
    "for abstract matrices (give explicit entries and the end time for a numerical solution)"
)


def _tau() -> sp.Dummy:
    return sp.Dummy("tau")


def _generator_linear_parts(p: Problem) -> dict[str, sp.Expr] | None:
    """name -> scalar coefficient when the generator is a sum of scalar functions times matrices."""
    parts: dict[str, sp.Expr] = {}
    for term in p.terms:
        if len(term.factors) != 1:
            return None
        f = term.factors[0]
        if not isinstance(f, Sym) or f.power != 1 or f.deriv:
            return None
        parts[f.name] = sp.expand(parts.get(f.name, sp.Integer(0)) + term.coeff)
    return parts


def _square(p: Problem) -> Attempt | None:
    if p.shape[0] != p.shape[1]:
        return Attempt(unsupported="the generator must be a square matrix")
    return None


def _initial_condition(p: Problem, u: Callable) -> Count:  # type: ignore[type-arg]
    """U(t0) = I at sampled parameters."""
    assert p.var is not None and p.lower is not None  # noqa: S101
    var, lower = p.var, p.lower

    def check(env) -> bool:  # type: ignore[no-untyped-def]
        start = p.ev.scalar(lower, env.scalars)
        return close(u(env.at(var, start)), mp.eye(p.shape[0]))

    return numeric_points(p.setup, p.ev, check, deadline=p.deadline, max_points=4)


def _evolution_equation(p: Problem, u: Callable) -> tuple[Count, list[str]]:  # type: ignore[type-arg]
    """dU/dt = A U by a central difference of U against A U."""
    assert p.var is not None  # noqa: S101
    size = p.shape[0]
    return check_derivative(
        p.setup,
        p.ev,
        p.var,
        u,
        lambda env: p.ev.terms(p.terms, env, size) * u(env),
        deadline=p.deadline,
    )


def _merge(a: Count, b: Count) -> Count:
    return Count(a.checked + b.checked, a.agreed + b.agreed)


# -- closed forms --------------------------------------------------------------------------------
def run_evolution_commuting(p: Problem) -> Attempt:
    assert p.var is not None and p.lower is not None  # noqa: S101
    if (bad := _square(p)) is not None:
        return bad
    names = matrices_in(p.terms)
    if names <= set(p.explicit):
        return _explicit_commuting(p)
    return _abstract_commuting(p, names)


def _evolution_steps(p: Problem, result: str, note: str) -> list[MatrixStep]:
    assert p.var is not None  # noqa: S101
    return [
        MatrixStep(
            method="evolution_commuting",
            title="Evolution operator of a generator that commutes with itself at all times",
            before_latex=(
                f"\\frac{{dU}}{{d{to_latex(p.var)}}} = \\left({render_terms(p.terms)}\\right) U,"
                f"\\quad U({to_latex(p.lower)}) = I"
            ),
            after_latex=result,
            note=note,
        )
    ]


def _abstract_commuting(p: Problem, names: set[str]) -> Attempt:
    assert p.var is not None and p.lower is not None  # noqa: S101
    var, lower = p.var, p.lower
    parts = _generator_linear_parts(p)
    if parts is None:
        return Attempt(
            unsupported=(
                "for abstract matrices the generator must be a sum of scalar functions of the "
                "variable times single matrices; " + NOT_CLOSED_FORM
            )
        )
    moving = sorted(n for n in parts if n in p.setup.dependent)
    if moving:
        return _dependent_representation(p, parts, moving)
    blocked: list[MatrixBlocked] = []
    needs = []
    for a, b in itertools.combinations(sorted(parts), 2):
        if not p.comm.commutes(a, b):
            needs.append(f"{a}{b} = {b}{a}")
    if needs:
        blocked.append(
            MatrixBlocked(
                method="evolution_commuting",
                title="U(t) = exp of the integral of the generator",
                needs=" and ".join(needs) + " (not established)",
                latex=render_terms(p.terms),
            )
        )
        return Attempt(blocked=blocked, unsupported=NOT_CLOSED_FORM)
    tau = _tau()
    arg: list[tuple[str, sp.Expr]] = []
    for name, c in sorted(parts.items()):
        integral = sp.integrate(c.subs(var, tau), (tau, lower, var))
        if integral.has(sp.Integral):
            return Attempt(
                unsupported="a scalar coefficient of the generator has no closed integral"
            )
        integral = sp.expand(integral)
        if integral != 0:
            arg.append((name, integral))
    if not arg:
        return Attempt(unsupported="the generator is zero")
    u_terms: Terms = (Term(sp.Integer(1), (Exp(tuple(arg)),)),)
    size = p.shape[0]
    count, directions = _evolution_equation(p, lambda env: p.ev.terms(u_terms, env, size))
    count = _merge(count, _initial_condition(p, lambda env: p.ev.terms(u_terms, env, size)))
    symbolic = _series_leg(p, u_terms)
    result = render_terms(u_terms)
    assumptions = [f"{a}{b} = {b}{a}" for a, b in itertools.combinations(sorted(parts), 2)]
    return Attempt(
        Outcome(
            "expression",
            result,
            count,
            _EVOLUTION_NUMERIC.format(directions=" and ".join(directions)),
            steps=_evolution_steps(
                p, result, "U(t) = exp of the integral of A(t), valid because the matrices commute"
            ),
            assumptions=assumptions,
            symbolic=symbolic,
            symbolic_method=_SERIES_EVOLUTION,
        )
    )


_EVOLUTION_NUMERIC = (
    "central difference (step 1e-12, 25 digits; {directions} direction) of U(t) = exp(...) by "
    "mpmath.expm on random complex matrices, against A(t) U(t); and U(t0) = I at sampled parameters"
)
_SERIES_EVOLUTION = (
    "formal power series through total degree 4 modulo the declared commutations: the derivative of "
    "the series of U equals the series of A times the series of U, and U(t0) = I; exact, but not a "
    "proof for all degrees"
)


def _series_leg(p: Problem, u_terms: Terms) -> bool | None:
    assert p.var is not None and p.lower is not None  # noqa: S101
    try:
        u = series_of_terms(u_terms)
        a = series_of_terms(p.terms)
        derivative_ok = poly_equal(poly_diff(u, p.var), poly_mul(a, u), p.comm)
        initial_ok = poly_equal(poly_subs(u, p.var, p.lower), {(): sp.Integer(1)}, p.comm)
    except ValueError:
        return None
    return derivative_ok and initial_ok


def _dependent_representation(p: Problem, parts: dict[str, sp.Expr], moving: list[str]) -> Attempt:
    """exp(integral of c(tau) A(tau) dtau) for one time-dependent matrix that commutes with itself."""
    assert p.var is not None and p.lower is not None  # noqa: S101
    var, lower = p.var, p.lower
    if len(parts) != 1:
        return Attempt(
            unsupported=(
                "a time-dependent matrix is supported alone (times a scalar function of the "
                "variable); sums with other matrices need the commutation of the matrices at all "
                "times, which is not supported for abstract matrices"
            )
        )
    name = moving[0]
    if name not in p.comm.self_commuting:
        return Attempt(
            blocked=[
                MatrixBlocked(
                    method="evolution_commuting",
                    title="U(t) = exp of the integral of the generator",
                    needs=f"{name}(t1) {name}(t2) = {name}(t2) {name}(t1) at all times (not "
                    "established)",
                    latex=render_terms(p.terms),
                )
            ],
            unsupported=NOT_CLOSED_FORM,
        )
    c = parts[name]
    tau = _tau()
    size = p.shape[0]
    weights: list[sp.Expr] = []
    for k in range(3):  # A(t) = C0 + t C1 + t^2 C2 in the numeric leg
        w = sp.integrate(sp.expand(c.subs(var, tau) * tau**k), (tau, lower, var))
        if w.has(sp.Integral):
            return Attempt(
                unsupported="the scalar coefficient of the generator has no closed integral"
            )
        weights.append(sp.expand(w))

    def u(env):  # type: ignore[no-untyped-def]
        total = mp.zeros(size, size)
        for k, w in enumerate(weights):
            total = total + p.ev.scalar(w, env.scalars) * env.poly[name][k]
        return mp.expm(total)

    count, directions = _evolution_equation(p, u)
    count = _merge(count, _initial_condition(p, u))
    coefficient = "" if c == 1 else f"{to_latex(c.subs(var, tau))}\\,"
    result = (
        f"\\exp\\left(\\int_{{{to_latex(lower)}}}^{{{to_latex(var)}}} {coefficient}"
        f"{{{_name(name)}}}({to_latex(tau)})\\,d{to_latex(tau)}\\right)"
    )
    return Attempt(
        Outcome(
            "representation",
            result,
            count,
            _EVOLUTION_NUMERIC.format(directions=" and ".join(directions))
            + " (the time-dependent matrix is sampled as a quadratic polynomial in the variable "
            "whose values commute at all times; its integral is computed exactly)",
            evaluated=False,
            steps=_evolution_steps(
                p,
                result,
                f"{name}(t1) {name}(t2) = {name}(t2) {name}(t1) at all times (declared or from "
                "the entries)",
            ),
            assumptions=[f"{name}(t1) {name}(t2) = {name}(t2) {name}(t1) at all times"],
            symbolic=None,
            notes=[
                "the integral of a matrix function is written, not evaluated: this is an exact "
                "representation"
            ],
        )
    )


def _name(name: str) -> str:
    return str(to_latex(sp.Symbol(name)))


def _explicit_commuting(p: Problem) -> Attempt:
    assert p.var is not None and p.lower is not None  # noqa: S101
    var, lower = p.var, p.lower
    size = p.shape[0]
    try:
        with symbolic_budget(p):
            generator = explicit_value(p.terms, p.explicit, size)
            tau = _tau()
            shifted = generator.subs(var, tau)
            commutator = (generator * shifted - shifted * generator).applyfunc(sp.simplify)
            if commutator != sp.zeros(size, size):
                return _numeric_or_blocked(p)
            integral = generator.subs(var, tau).applyfunc(
                lambda e: sp.integrate(e, (tau, lower, var))
            )
            if any(e.has(sp.Integral) for e in integral):
                return _numeric_or_blocked(p)
            value = _simplified(integral.exp())
    except (NotImplementedError, ValueError, sp.matrices.exceptions.MatrixError, SymbolicTimeout):
        return _numeric_or_blocked(p)
    candidate = _numeric_candidate(p, value)
    count, directions = _evolution_equation(p, candidate)
    count = _merge(count, _initial_condition(p, candidate))
    with symbolic_budget(p):
        try:
            residual = (value.diff(var) - generator * value).applyfunc(sp.simplify)
            at_start = value.subs(var, lower).applyfunc(sp.simplify)
            exact: bool | None = (
                True if residual == sp.zeros(size, size) and at_start == sp.eye(size) else None
            )
        except SymbolicTimeout:
            exact = None
    latex = str(to_latex(value))
    return Attempt(
        Outcome(
            "explicit",
            latex,
            count,
            _EVOLUTION_NUMERIC.format(directions=" and ".join(directions)),
            entries=_entries(value),
            steps=_evolution_steps(
                p,
                latex,
                "U(t) = exp of the integral of A(t); the entries satisfy A(t1) A(t2) = A(t2) A(t1) "
                "for all times (checked symbolically)",
            ),
            assumptions=["A(t1) A(t2) = A(t2) A(t1) at all times (from the entries)"],
            symbolic=exact,
            symbolic_method=(
                "the exact derivative minus A U and U(t0) - I simplify to zero entry by entry "
                "(SymPy); an unsimplified residual is inconclusive, not a refutation"
            ),
        )
    )


def _numeric_or_blocked(p: Problem) -> Attempt:
    return Attempt(
        blocked=[
            MatrixBlocked(
                method="evolution_commuting",
                title="U(t) = exp of the integral of the generator",
                needs="A(t1) A(t2) = A(t2) A(t1) at all times (not established for these entries)",
                latex=render_terms(p.terms),
            )
        ]
    )


# -- numerical solution ----------------------------------------------------------------------------
Vector = list[complex]


class _Stop(Exception):
    """The numerical solution cannot continue; the message says why."""


def _matmul(a: list[list[complex]], b: list[list[complex]]) -> list[list[complex]]:
    n, m, k = len(a), len(b[0]), len(b)
    return [[sum(a[i][r] * b[r][j] for r in range(k)) for j in range(m)] for i in range(n)]


_A = (
    (),
    (1 / 5,),
    (3 / 40, 9 / 40),
    (44 / 45, -56 / 15, 32 / 9),
    (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
    (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
    (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84),
)
_C = (0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0)
_E = (71 / 57600, 0.0, -71 / 16695, 71 / 1920, -17253 / 339200, 22 / 525, -1 / 40)


class _Field:
    """The vector field y -> A(t) y for the n columns of U, evaluated in double precision."""

    def __init__(self, generator: Callable[[float], list[list[complex]]], n: int) -> None:
        self.generator = generator
        self.n = n

    def __call__(self, t: float, y: Vector) -> Vector:
        n = self.n
        g = self.generator(t)
        u = [[y[i * n + j] for j in range(n)] for i in range(n)]
        out = _matmul(g, u)
        flat = [out[i][j] for i in range(n) for j in range(n)]
        if any(not (cmath.isfinite(x)) for x in flat):
            raise _Stop("the generator becomes infinite along the span (a singularity)")
        return flat


def _dopri(
    fn: _Field, y0: Vector, times: list[float], deadline: float
) -> tuple[list[Vector], list[float]]:
    """Adaptive Dormand-Prince 5(4) through `times`; also the smallest accepted step per segment."""
    n = len(y0)
    direction = 1.0 if times[-1] >= times[0] else -1.0
    t = times[0]
    y = list(y0)
    states = [list(y)]
    k1 = fn(t, y)
    h = abs(times[1] - times[0]) / 10
    attempts = 0
    smallest: list[float] = []
    for target in times[1:]:
        smallest.append(abs(target - t))
        while (target - t) * direction > 0:
            attempts += 1
            if attempts > MAX_ATTEMPTS:
                raise LimitError("Too many integration steps were needed (the system is stiff).")
            if time.monotonic() > deadline:
                raise LimitError("The integration used up its time budget.")
            hit = h >= abs(target - t)
            step = (abs(target - t) if hit else h) * direction
            ks = [k1]
            ys = y
            for stage in range(1, 7):
                ys = [
                    y[i] + step * sum(a * k[i] for a, k in zip(_A[stage], ks, strict=False) if a)
                    for i in range(n)
                ]
                ks.append(fn(t + _C[stage] * step, ys))
            err = [step * sum(e * k[i] for e, k in zip(_E, ks, strict=True) if e) for i in range(n)]
            norm = math.sqrt(
                sum(
                    (abs(err[i]) / (ATOL + RTOL * max(abs(y[i]), abs(ys[i])))) ** 2
                    for i in range(n)
                )
                / n
            )
            factor = 5.0 if norm == 0 else min(5.0, max(0.2, 0.9 * norm ** (-0.2)))
            if norm <= 1.0:
                t = target if hit else t + step
                y = ys
                k1 = ks[6]
                if not hit:
                    smallest[-1] = min(smallest[-1], abs(step))
                h = max(h, abs(step) * factor) if hit and factor >= 1 else abs(step) * factor
            else:
                h = abs(step) * factor
            if h < 1e-13 * max(1.0, abs(t)):
                raise _Stop("the step size collapsed: a singularity or a stiff system")
        states.append(list(y))
    return states, smallest


def _rk4(
    fn: _Field, y0: Vector, times: list[float], counts: list[int], deadline: float
) -> list[Vector]:
    n = len(y0)
    y = list(y0)
    states = [list(y)]
    for (a, b), count in zip(itertools.pairwise(times), counts, strict=True):
        h = (b - a) / count
        t = a
        for _ in range(count):
            k1 = fn(t, y)
            k2 = fn(t + h / 2, [y[i] + h / 2 * k1[i] for i in range(n)])
            k3 = fn(t + h / 2, [y[i] + h / 2 * k2[i] for i in range(n)])
            k4 = fn(t + h, [y[i] + h * k3[i] for i in range(n)])
            y = [y[i] + h / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(n)]
            t += h
        if time.monotonic() > deadline:
            raise LimitError("The cross-check used up its time budget.")
        states.append(list(y))
    return states


def _distance(a: Vector, b: Vector) -> float:
    return max((abs(x - y) for x, y in zip(a, b, strict=True)), default=0.0) / max(
        1.0, max(abs(x) for x in a), max(abs(y) for y in b)
    )


def _format(z: complex) -> str:
    re, im = z.real, z.imag
    scale = max(abs(re), abs(im), 1.0)
    if abs(im) < 1e-13 * scale:
        return f"{re:.12g}"
    if abs(re) < 1e-13 * scale:
        return f"{im:.12g} i"
    return f"{re:.12g} {'+' if im >= 0 else '-'} {abs(im):.12g} i"


def _latex_matrix(rows: list[list[str]]) -> str:
    cells = [" & ".join(c.replace(" i", r"\,\mathrm{i}") for c in row) for row in rows]
    return "\\begin{pmatrix} " + " \\\\ ".join(cells) + " \\end{pmatrix}"


def run_evolution_numeric(p: Problem) -> Attempt:
    assert p.var is not None and p.lower is not None  # noqa: S101
    if (bad := _square(p)) is not None:
        return bad
    var = p.var
    names = matrices_in(p.terms)
    n = p.shape[0]
    if not names <= set(p.explicit):
        return Attempt(unsupported=NOT_CLOSED_FORM)
    if n > MAX_SIZE:
        return Attempt(unsupported=f"the numerical solution is limited to {MAX_SIZE} by {MAX_SIZE}")
    if any(isinstance(f, Exp) for t in p.terms for f in t.factors):
        return Attempt(unsupported="the generator must be a polynomial in the matrices")
    if p.upper is None:
        return Attempt(
            unsupported=(
                "the generator does not commute with itself at different times: give the end "
                "time of the span for a numerical solution (it is a number only you can choose)"
            )
        )
    try:
        with symbolic_budget(p):
            generator = explicit_value(p.terms, p.explicit, n)
    except (NotImplementedError, ValueError, SymbolicTimeout):
        return Attempt(unsupported="the generator could not be formed from the entries")
    extra = generator.free_symbols - {var}
    if extra or not (p.lower.is_number and p.upper.is_number):
        return Attempt(
            unsupported=(
                "the numerical solution needs numbers for every parameter and for both ends of the "
                "span; give values for: "
                + ", ".join(
                    sorted(s.name for s in extra | (p.lower.free_symbols | p.upper.free_symbols))
                )
            )
        )
    t0, t1 = float(p.lower), float(p.upper)
    if t0 == t1:
        return Attempt(unsupported="the span is empty")
    rows = generator.tolist()
    fn_entries = sp.lambdify(var, rows, modules="cmath")

    def gen(t: float) -> list[list[complex]]:
        return [[complex(x) for x in row] for row in fn_entries(complex(t))]

    field = _Field(gen, n)
    identity = [complex(1.0 if i == j else 0.0) for i in range(n) for j in range(n)]
    times = [t0 + (t1 - t0) * k / SAMPLES for k in range(SAMPLES + 1)]
    started = time.monotonic()
    try:
        states, smallest = _dopri(field, identity, times, p.deadline)
        counts = [
            min(MAX_RK4_STEPS, max(MIN_RK4_STEPS, math.ceil(8 * abs(b - a) / s)))
            for (a, b), s in zip(itertools.pairwise(times), smallest, strict=True)
        ]
        second = _rk4(field, identity, times, counts, p.deadline)
        backward, _ = _dopri(field, states[-1], [t1, t0], p.deadline)
    except _Stop as exc:
        return Attempt(unsupported=f"the numerical solution stopped: {exc}")
    elapsed_integrate = time.monotonic() - started

    count = Count()
    worst = 0.0
    for a, b in zip(states, second, strict=True):  # (1) two different methods agree
        d = _distance(a, b)
        worst = max(worst, d)
        count.checked += 1
        count.agreed += d <= METHOD_TOL
    d = _distance(backward[-1], identity)  # (2) integrating back recovers the identity
    worst = max(worst, d)
    count.checked += 1
    count.agreed += d <= REVERSAL_TOL
    ok_liouville, liouville_error = _liouville(generator, var, p.lower, p.upper, states[-1], n)
    worst = max(worst, liouville_error)
    record = telemetry()
    if record is not None:
        record.numeric_s += elapsed_integrate
        record.digits = 15  # double precision
        record.note_residual(worst, max(METHOD_TOL, REVERSAL_TOL, LIOUVILLE_TOL))
    count.checked += 1  # (3) det U = exp(integral of trace A), by quadrature
    count.agreed += ok_liouville
    count.checked += 1  # (4) the initial condition holds exactly
    count.agreed += states[0] == identity

    table = []
    steps = [
        MatrixStep(
            method="evolution_numeric",
            title="Numerical solution of dU/dt = A(t) U, U(t0) = I",
            before_latex=(
                f"\\frac{{dU}}{{d{to_latex(var)}}} = \\left({render_terms(p.terms)}\\right) U"
            ),
            after_latex=None,
            note=(
                f"Dormand-Prince 5(4) (rtol {RTOL:g}, atol {ATOL:g}) on [{t0:g}, {t1:g}], "
                "double precision; a measurement on this span, not a closed form"
            ),
        )
    ]
    for k in range(1, len(states)):
        cells = [[_format(states[k][i * n + j]) for j in range(n)] for i in range(n)]
        table.append(cells)
        steps.append(
            MatrixStep(
                method="evolution_numeric",
                title=f"U at t = {times[k]:g}",
                after_latex=_latex_matrix(cells),
            )
        )
    final = table[-1]
    return Attempt(
        Outcome(
            "numeric",
            _latex_matrix(final),
            count,
            "a second method (classical RK4, at least 8 times finer than the smallest accepted "
            f"step) agrees at all {SAMPLES + 1} sample times within {METHOD_TOL:g}; integrating "
            f"backwards from t1 recovers I within {REVERSAL_TOL:g}; det U = exp(integral of tr A) "
            f"by quadrature within {LIOUVILLE_TOL:g}",
            evaluated=False,
            entries=final,
            steps=steps,
            symbolic=None,
            notes=[
                f"numerical approximation on [{t0:g}, {t1:g}], double precision, largest "
                f"disagreement {worst:.1e}; U is shown at t = {t1:g}",
                "this is a measurement on that span, never a statement for all times; the "
                "generator does not commute with itself at different times, so there is no "
                "closed form of this kind",
            ],
            note=f"integration took {elapsed_integrate:.2f} s",
        )
    )


def _liouville(
    generator: sp.Matrix,
    var: sp.Symbol,
    lower: sp.Expr,
    upper: sp.Expr,
    final: Vector,
    n: int,
) -> tuple[bool, float]:
    """det U(t1) against exp of the integral of the trace, which uses a different code path."""
    trace = sp.lambdify(var, generator.trace(), modules="mpmath")
    with mp.workdps(30):
        integral = mp.quad(lambda x: trace(x), [mp.mpf(float(lower)), mp.mpf(float(upper))])
        want = mp.exp(integral)
        matrix = mp.matrix(n, n)
        for i in range(n):
            for j in range(n):
                matrix[i, j] = mp.mpc(final[i * n + j])
        have = mp.det(matrix)
        error = float(abs(have - want) / max(1, abs(want)))
    return error <= LIOUVILLE_TOL, error


def rk4_observed_order(
    generator: sp.Matrix, var: sp.Symbol, t0: float, t1: float, steps: int = 200
) -> tuple[list[float], float]:
    """Errors of fixed-step RK4 at N, 2N and 4N steps against the adaptive solution, and the
    observed order log2(e_N / e_2N). A measurement of convergence for the benchmark: for a smooth
    generator the order is close to 4."""
    n = generator.rows
    rows = generator.tolist()
    fn_entries = sp.lambdify(var, rows, modules="cmath")

    def gen(t: float) -> list[list[complex]]:
        return [[complex(x) for x in row] for row in fn_entries(complex(t))]

    field = _Field(gen, n)
    identity = [complex(1.0 if i == j else 0.0) for i in range(n) for j in range(n)]
    deadline = time.monotonic() + 60.0
    reference, _ = _dopri(field, identity, [t0, t1], deadline)
    errors = []
    for factor in (1, 2, 4):
        final = _rk4(field, identity, [t0, t1], [steps * factor], deadline)[-1]
        errors.append(_distance(final, reference[-1]))
    order = math.log2(errors[0] / errors[1]) if errors[0] > 0 and errors[1] > 0 else float("nan")
    return errors, order
