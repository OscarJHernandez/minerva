"""Isolating a variable in an equation, with every solution and every condition listed.

Two kinds of equation are covered; anything else is `unsupported` and never answered in part.

*The unknown occurs once.* The layers around it are peeled one at a time, each a recorded step that
leaves an equivalent equation under a stated condition: subtracting a term; dividing by a factor
(which must not be zero); taking an n-th root (all n roots, 2 <= n <= 6); a reciprocal (the other
side must not be zero); a principal root `g^(1/q) = h` (g = h^q, and h must lie in the principal
sector, which can make a candidate extraneous); `exp(g) = h` or `b^g = h` for b known positive
(g = log h + 2 pi i k, k any integer, h not zero); `log g = h` (g = exp h, Im h in the principal strip).

*The unknown occurs more than once.* The numerator of the combined equation must be a polynomial in
it. It is factored; linear and quadratic factors are solved in closed form with symbolic
coefficients, cubic and quartic factors only with rational coefficients, anything else is refused
(no closed form in general). Leading coefficients that may vanish are reported as conditions with the
exceptional case described; values that zero a denominator are removed (or made conditions).

Checks, none of which use the method that produced the answer:
- every solution is substituted into the *original* equation at sampled parameter values where its
  conditions hold (high precision);
- nothing was dropped: for the polynomial path the distinct numerical roots of the numerator at
  those parameter values must be exactly the solutions (less those the denominator removes); for
  peeling, the same peeling is run with the other side replaced by a free symbol, a random value of
  the unknown is pushed forward through the original expression, and some solution must return it.
Solutions are over the complex numbers. Declared assumptions on the unknown do not select among them.
"""

from __future__ import annotations

import cmath
import math
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import mpmath
import sympy as sp

from .claims import split_equation
from .errors import LimitError, MinervaMathError, ParseError, UnsupportedError
from .latexprint import to_latex
from .models import (
    Diagnostics,
    ExpressionInput,
    IsolateResult,
    Limits,
    Reason,
    Solution,
    Status,
    Step,
    Verification,
)
from .ops import _failure, parse_expression
from .verify import _POLE, MIN_VALID_POINTS, _evaluate_ex, _sample

MAX_ROOT_ORDER = 6
MAX_SOLUTIONS = 12
ATTEMPTS = 80
K_RANGE = range(-6, 7)
TOL = 1e-8
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,31}$")


# --- conditions ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Cond:
    """A condition under which a step is an equivalence. Decided from the assumptions when possible
    (`status`), otherwise reported, and tested numerically at sample points (`at`)."""

    kind: Literal["nonzero", "principal", "strip"]
    expr: sp.Expr
    order: int = 1

    def latex(self) -> str:
        shown = to_latex(self.expr)
        if self.kind == "nonzero":
            return rf"{shown} \neq 0"
        if self.kind == "principal":
            return rf"-\frac{{\pi}}{{{self.order}}} < \arg\left({shown}\right) \le \frac{{\pi}}{{{self.order}}}"
        return rf"-\pi < \operatorname{{Im}}\left({shown}\right) \le \pi"

    def status(self) -> bool | None:
        """True: provable from the declared assumptions. False: provably violated. None: unknown."""
        e = self.expr
        if self.kind == "nonzero":
            zero = e.is_zero
            return None if zero is None else not zero
        if self.kind == "principal":
            if e.is_nonnegative:
                return True
            if e.is_negative and self.order >= 2:
                return False
        elif e.is_real or _ratio_of_reals(e):
            return True
        if e.is_number and not e.free_symbols:
            return self._decide(complex(sp.N(e, 30)))
        return None

    def at(self, values: dict[sp.Symbol, sp.Expr]) -> bool | None:
        value = _evaluate_ex(self.expr, values)
        if value is _POLE or not isinstance(value, complex):
            return None
        return self._decide(value)

    def _decide(self, z: complex) -> bool | None:
        if self.kind == "nonzero":
            return abs(z) > 1e-24  # zero to working precision (values are evaluated with 30 digits)
        eps = 1e-9
        if self.kind == "principal":
            if abs(z) < eps:
                return True
            angle, edge = cmath.phase(z), math.pi / self.order
            if abs(angle - edge) < eps or abs(angle + edge) < eps:
                return None  # on the boundary: the numbers cannot tell
            return -edge < angle < edge
        edge = math.pi
        if abs(z.imag - edge) < eps or abs(z.imag + edge) < eps:
            return None
        return -edge < z.imag < edge


def _ratio_of_reals(expr: sp.Expr) -> bool:
    """A quotient of real quantities is real wherever it is defined. SymPy will not say so (the divisor
    might be zero), but every division in a peeled chain has already put `divisor != 0` among the
    conditions, so within the stated conditions the quotient is real."""
    numerator, denominator = sp.fraction(sp.together(expr))
    return bool(numerator.is_real and denominator.is_real)


def nonzero(expr: sp.Expr) -> list[Cond] | None:
    """The conditions for `expr != 0`, one per irreducible factor, without the ones the assumptions
    prove. None: `expr` is provably zero (so the step is impossible)."""
    try:
        numerator, denominator = sp.fraction(sp.together(expr))
        parts = [sp.factor_list(numerator), sp.factor_list(denominator)]
    except (sp.PolynomialError, ValueError):
        parts = [(sp.Integer(1), [(expr, 1)])]
    found: list[Cond] = []
    for coefficient, factors in parts:
        if coefficient == 0:
            return None
        for factor, _ in factors:
            if not factor.free_symbols:
                if factor == 0:
                    return None
                continue
            cond = Cond("nonzero", factor)
            decided = cond.status()
            if decided is False:
                return None
            if decided is None and not any(
                sp.expand(factor - c.expr) == 0 or sp.expand(factor + c.expr) == 0 for c in found
            ):
                found.append(cond)
    return found


def _unique(conds: list[Cond]) -> list[Cond]:
    out: list[Cond] = []
    for cond in conds:
        if not any(
            c.kind == cond.kind
            and c.order == cond.order
            and (
                c.expr == cond.expr
                or (cond.kind == "nonzero" and sp.expand(c.expr + cond.expr) == 0)
            )
            for c in out
        ):
            out.append(cond)
    return out


@dataclass
class State:
    """One branch while peeling: the value the inner expression must take, and conditions so far."""

    h: sp.Expr
    conds: list[Cond] = field(default_factory=list)


# --- peeling --------------------------------------------------------------------------------------------
def _split(G: sp.Expr, x: sp.Symbol) -> tuple[str, sp.Expr, sp.Expr]:
    if G.is_Add:
        inside = [a for a in G.args if a.has(x)]
        return "add", inside[0], sp.Add(*[a for a in G.args if not a.has(x)])
    if G.is_Mul:
        inside = [a for a in G.args if a.has(x)]
        return "mul", inside[0], sp.Mul(*[a for a in G.args if not a.has(x)])
    if G.is_Pow:
        base, exponent = G.args
        if base.has(x) and not exponent.has(x):
            return "power", base, exponent
        if exponent.has(x) and not base.has(x):
            return "exponential", exponent, base
    if isinstance(G, sp.exp):
        return "exp", G.args[0], sp.Integer(0)
    if isinstance(G, sp.log) and len(G.args) == 1:
        return "log", G.args[0], sp.Integer(0)
    raise UnsupportedError(
        "the unknown is inside a form this version cannot invert (only sums, products, powers, "
        "roots, exp and log around it are supported; periodic functions are not)"
    )


def _roots_of(h: sp.Expr, n: int) -> list[sp.Expr]:
    if n > MAX_ROOT_ORDER:
        raise UnsupportedError(f"roots of order above {MAX_ROOT_ORDER} are not supported")
    principal = h ** sp.Rational(1, n)
    return [
        principal * (sp.cos(2 * sp.pi * j / n) + sp.I * sp.sin(2 * sp.pi * j / n)) for j in range(n)
    ]


def _fresh_integer(taken: set[str]) -> sp.Symbol:
    index = 1
    while f"k_{index}" in taken:
        index += 1
    taken.add(f"k_{index}")
    return sp.Symbol(f"k_{index}", integer=True)


def _with(state: State, h: sp.Expr, *conds: Cond) -> State | None:
    """The state after a step, or None if one of its conditions is provably violated."""
    out = list(state.conds)
    for cond in conds:
        decided = cond.status()
        if decided is False:
            return None
        if decided is None:
            out.append(cond)
    return State(h, _unique(out))


def _nonzero_state(state: State, h: sp.Expr, requires: sp.Expr) -> State | None:
    needed = nonzero(requires)
    if needed is None:
        return None
    return _with(state, h, *needed)


def _apply_layer(
    kind: str, extra: sp.Expr, state: State, taken: set[str], integer: list[sp.Symbol]
) -> tuple[list[State | None], str]:
    h = state.h
    if kind == "add":
        return [_with(state, h - extra)], "subtract"
    if kind == "mul":
        return [_nonzero_state(state, h / extra, extra)], "divide"
    if kind == "power":
        if extra.is_Integer and extra >= 2:
            return [_with(state, root) for root in _roots_of(h, int(extra))], "root"
        if extra.is_Integer and extra <= -1:
            flipped = _nonzero_state(state, 1 / h, h)
            if flipped is None:
                return [None], "reciprocal"
            if extra == -1:
                return [flipped], "reciprocal"
            return [
                _with(flipped, root) for root in _roots_of(flipped.h, int(-extra))
            ], "reciprocal"
        if extra.is_Rational and extra.p == 1 and 2 <= extra.q <= MAX_ROOT_ORDER:
            q = int(extra.q)
            return [_with(state, h**q, Cond("principal", h, q))], "power"
        raise UnsupportedError(
            "only integer powers and principal roots of the unknown are supported"
        )
    if kind == "exponential":
        if extra.is_positive is not True:
            raise UnsupportedError(
                "the unknown is in an exponent whose base is not known to be positive; declare it"
            )
        k = _fresh_integer(taken)
        integer.append(k)
        guard = _nonzero_state(state, (sp.log(h) + 2 * sp.pi * sp.I * k) / sp.log(extra), h)
        if guard is None:
            return [None], "logarithm"
        return [_nonzero_state(guard, guard.h, extra - 1)], "logarithm"
    if kind == "exp":
        k = _fresh_integer(taken)
        integer.append(k)
        return [_nonzero_state(state, sp.log(h) + 2 * sp.pi * sp.I * k, h)], "logarithm"
    return [_with(state, sp.exp(h), Cond("strip", h))], "exponential"  # kind == "log"


@dataclass
class Peeled:
    states: list[State]
    steps: list[Step]
    dropped: int
    integers: list[sp.Symbol]


def peel(G: sp.Expr, h: sp.Expr, x: sp.Symbol, taken: set[str]) -> Peeled:
    states = [State(h)]
    steps: list[Step] = []
    integers: list[sp.Symbol] = []
    dropped = 0
    while x != G and states:
        kind, inner, extra = _split(G, x)
        before = f"{to_latex(G)} = {to_latex(states[0].h)}"
        produced: list[State] = []
        name = kind
        for state in states:
            results, name = _apply_layer(kind, extra, state, taken, integers)
            for result in results:
                if result is None:
                    dropped += 1
                else:
                    produced.append(result)
        if len(produced) > MAX_SOLUTIONS:
            raise LimitError("too many branches")
        shown = _distinct(produced)
        new_conditions = [
            c.latex() for c in (shown[0].conds if shown else []) if c not in states[0].conds
        ]
        steps.append(
            Step(
                action="peel",
                name=name,
                before_latex=before,
                after_latex=r" \quad\text{or}\quad ".join(
                    f"{to_latex(inner)} = {to_latex(s.h)}" for s in shown[:4]
                )
                + (r" \quad\text{or}\ \dots" if len(shown) > 4 else ""),
                conditions_latex=new_conditions,
            )
        )
        states, G = produced, inner
    return Peeled(_distinct(states), steps, dropped, integers)


def _distinct(states: list[State]) -> list[State]:
    out: list[State] = []
    for state in states:
        if not any(sp.expand(state.h - other.h) == 0 for other in out):
            out.append(state)
    return out


# --- polynomial and rational equations -------------------------------------------------------------------
@dataclass
class Solved:
    states: list[State]
    steps: list[Step]
    dropped: int
    global_conds: list[Cond]
    excluded: list[str]
    numerator: sp.Expr
    denominator: sp.Expr


def _closed_roots(factor: sp.Poly, x: sp.Symbol) -> tuple[list[sp.Expr], list[Cond]]:
    degree = factor.degree()
    coefficients = factor.all_coeffs()
    lead = coefficients[0]
    conds = nonzero(lead)
    if conds is None:
        return [], []
    if degree == 1:
        return [-coefficients[1] / lead], conds
    if degree == 2:
        _, b, c = coefficients
        root = sp.sqrt(b**2 - 4 * lead * c)
        return [(-b + root) / (2 * lead), (-b - root) / (2 * lead)], conds
    if (
        degree in (3, 4)
        and not factor.free_symbols - {x}
        and (factor.domain.is_QQ or factor.domain.is_ZZ)
    ):
        found = sp.roots(factor.as_expr(), x, cubics=True, quartics=True)
        if sum(found.values()) == degree:
            return list(found), conds
    raise UnsupportedError(
        "no closed form is produced for a factor of this degree with these coefficients: a general "
        "polynomial of degree 5 or more has none (Abel-Ruffini), and cubics and quartics are solved "
        "here only with rational coefficients"
    )


def solve_rational(L: sp.Expr, R: sp.Expr, x: sp.Symbol, recorded: list[sp.Expr]) -> Solved:
    numerator, denominator = sp.fraction(sp.together(L - R))
    try:
        poly = sp.Poly(sp.expand(numerator), x)
    except sp.PolynomialError as exc:
        raise UnsupportedError("the equation is not polynomial or rational in the unknown") from exc
    if poly.is_zero:
        raise UnsupportedError(
            "the equation holds for every value of the unknown where it is defined"
        )
    steps = [
        Step(
            action="polynomial",
            name="common denominator",
            before_latex=f"{to_latex(L)} = {to_latex(R)}",
            after_latex=f"{to_latex(sp.expand(numerator))} = 0",
            note="multiplied by the common denominator",
        )
    ]
    global_conds: list[Cond] = []
    states: list[State] = []
    if poly.degree() >= 1:
        content, factors = sp.factor_list(poly.as_expr(), x)
        needed = nonzero(content)
        if needed is None:
            return Solved([], steps, 0, [], [], numerator, denominator)
        global_conds.extend(needed)
        for factor, _ in factors:
            fp = sp.Poly(factor, x)
            if fp.degree() < 1:
                continue
            roots, conds = _closed_roots(fp, x)
            states.extend(State(r, list(conds)) for r in roots)
            steps.append(
                Step(
                    action="polynomial",
                    name=f"degree {fp.degree()}",
                    before_latex=f"{to_latex(factor)} = 0",
                    after_latex=r" \quad\text{or}\quad ".join(
                        f"{x.name} = {to_latex(r)}" for r in roots
                    ),
                    conditions_latex=[c.latex() for c in conds],
                )
            )
    # Values that make a denominator vanish are not solutions of the original equation.
    pole_sources = [denominator, *[d for d in recorded if d.has(x)]]
    excluded: list[str] = []
    for source in pole_sources:
        for cond in nonzero(source) or []:
            if cond.expr.has(x) and cond.latex() not in excluded:
                excluded.append(cond.latex())
    dropped = 0
    kept: list[State] = []
    for state in states:
        keep = True
        extra: list[Cond] = []
        for source in pole_sources:
            at_root = sp.simplify(source.xreplace({x: state.h}))
            needed = nonzero(at_root)
            if needed is None:
                keep = False
                break
            extra.extend(needed)
        if keep:
            kept.append(State(state.h, _unique([*state.conds, *extra])))
        else:
            dropped += 1
    return Solved(_distinct(kept), steps, dropped, global_conds, excluded, numerator, denominator)


# --- checking ------------------------------------------------------------------------------------------------
def _env(
    params: list[sp.Symbol], integers: list[sp.Symbol], rng: random.Random
) -> dict[sp.Symbol, sp.Expr]:
    env: dict[sp.Symbol, sp.Expr] = {s: _sample(s, rng, spread=max(1, len(params))) for s in params}
    for k in integers:
        env[k] = sp.Integer(rng.randint(-2, 2))
    return env


def _holds(conds: list[Cond], env: dict[sp.Symbol, sp.Expr]) -> bool:
    """All conditions certainly hold at this point (unknown counts as not holding: the point is skipped)."""
    return all(c.at(env) is True for c in conds)


def back_substitute(
    L: sp.Expr,
    R: sp.Expr,
    x: sp.Symbol,
    states: list[State],
    params: list[sp.Symbol],
    integers: list[sp.Symbol],
    rng: random.Random,
) -> tuple[int, int, bool]:
    """Substitute each solution into the original equation where its conditions hold."""
    checked = agreed = 0
    enough = True
    for state in states:
        mine = 0
        for _ in range(ATTEMPTS):
            if mine >= 6:
                break
            env = _env(params, integers, rng)
            if not _holds(state.conds, env):
                continue
            value = state.h.xreplace(env)
            if not isinstance(_evaluate_ex(value, {}), complex):
                continue
            left = _evaluate_ex(L, {**env, x: value})
            right = _evaluate_ex(R, {**env, x: value})
            if not isinstance(left, complex) or not isinstance(right, complex):
                continue
            mine += 1
            checked += 1
            agreed += int(abs(left - right) <= TOL * max(1.0, abs(left), abs(right)))
        enough = enough and mine >= MIN_VALID_POINTS
    return checked, agreed, enough


def forward_probe(
    G: sp.Expr,
    x: sp.Symbol,
    placeholder: sp.Dummy,
    params: list[sp.Symbol],
    peeled: Peeled,
    rng: random.Random,
) -> tuple[int, int]:
    """Push a random value of the unknown forward through the expression; some solution must return it."""
    checked = hit = 0
    for _ in range(ATTEMPTS):
        if checked >= 8:
            break
        values = {s: _sample(s, rng, spread=max(1, len(params))) for s in params}
        # Alternate ordinary and wide values: only wide ones leave the principal branch of log, which is
        # what shows whether the integer families (and every root) are really all there.
        x0 = _sample(x, rng, wide=bool(checked % 2))
        pushed = _evaluate_ex(G, {**values, x: x0})
        if not isinstance(pushed, complex) or not 1e-9 < abs(pushed) < 1e30:
            continue  # too extreme to tell a solution from rounding: not a probe
        base = {**values, placeholder: sp.Float(pushed.real, 30) + sp.I * sp.Float(pushed.imag, 30)}
        checked += 1
        target = complex(x0)
        for state in peeled.states:
            if any(
                _returns(state, {**base, **ks}, target)
                for ks in _k_candidates(state, base, peeled.integers, target)
            ):
                hit += 1
                break
    return checked, hit


def _returns(state: State, env: dict[sp.Symbol, sp.Expr], target: complex) -> bool:
    if not _holds(state.conds, env):
        return False
    candidate = _evaluate_ex(state.h, env)
    return isinstance(candidate, complex) and abs(candidate - target) <= TOL * max(
        1.0, abs(candidate)
    )


def _k_candidates(
    state: State, base: dict[sp.Symbol, sp.Expr], integers: list[sp.Symbol], target: complex
) -> list[dict[sp.Symbol, sp.Expr]]:
    """Integer assignments worth trying. The value depends on the first integer through a log, so
    along that integer it is (nearly) a straight line: two evaluations give the integer that would
    hit `target`, and its neighbours are tried. A small fixed range covers the rest."""
    used = [k for k in integers if state.h.has(k)]
    if not used:
        return [{}]
    first, others = used[0], used[1:]
    guesses = set(K_RANGE)
    v0 = _evaluate_ex(state.h, {**base, first: sp.Integer(0), **{k: sp.Integer(0) for k in others}})
    v1 = _evaluate_ex(state.h, {**base, first: sp.Integer(1), **{k: sp.Integer(0) for k in others}})
    if isinstance(v0, complex) and isinstance(v1, complex) and v1 != v0:
        estimate = (target - v0) / (v1 - v0)
        centre = round(estimate.real) if abs(estimate.real) < 1e6 else 0
        guesses |= {centre - 1, centre, centre + 1}
    out: list[dict[sp.Symbol, sp.Expr]] = []
    for n in sorted(guesses):
        out.append({first: sp.Integer(n), **{k: sp.Integer(0) for k in others}})
    for k in others:
        out.extend({first: sp.Integer(n), k: sp.Integer(m)} for n in (0, 1) for m in range(-2, 3))
    return out


def _numeric_roots(
    numerator: sp.Expr, x: sp.Symbol, env: dict[sp.Symbol, sp.Expr], degree: int
) -> list[complex] | None:
    poly = sp.Poly(sp.expand(numerator.xreplace(env)), x)
    if poly.degree() != degree:
        return None
    try:
        with mpmath.workdps(40):
            roots = mpmath.polyroots(
                [complex(c) for c in poly.all_coeffs()], maxsteps=400, extraprec=200
            )
    except (mpmath.libmp.NoConvergence, ZeroDivisionError, ValueError, TypeError):
        return None
    return [complex(r) for r in roots]


def _same_distinct(a: list[complex], b: list[complex]) -> bool:
    def squash(values: list[complex]) -> list[complex]:
        out: list[complex] = []
        for v in values:
            if not any(abs(v - w) <= 1e-6 * max(1.0, abs(v)) for w in out):
                out.append(v)
        return out

    left, right = squash(a), squash(b)
    return len(left) == len(right) and all(
        any(abs(v - w) <= 1e-6 * max(1.0, abs(v)) for w in right) for v in left
    )


def count_roots(
    solved: Solved,
    x: sp.Symbol,
    params: list[sp.Symbol],
    rng: random.Random,
) -> tuple[int, int]:
    """At sampled parameter values the distinct roots of the numerator, less those the denominator
    removes, must be exactly the solutions whose conditions hold."""
    degree = sp.Poly(sp.expand(solved.numerator), x).degree()
    checked = agreed = 0
    for _ in range(ATTEMPTS):
        if checked >= 8:
            break
        env = _env(params, [], rng)
        numeric = _numeric_roots(solved.numerator, x, env, degree) if degree >= 1 else []
        if numeric is None:
            continue
        removed = []
        for r in numeric:
            denominator = _evaluate_ex(
                solved.denominator, {**env, x: sp.Float(r.real, 30) + sp.I * sp.Float(r.imag, 30)}
            )
            if denominator is _POLE or (
                isinstance(denominator, complex) and abs(denominator) < 1e-7
            ):
                removed.append(r)
        wanted = [r for r in numeric if r not in removed]
        ours: list[complex] = []
        usable = True
        for state in solved.states:
            decided = [c.at(env) for c in state.conds]
            if any(d is None for d in decided):
                usable = False
                break
            value = _evaluate_ex(state.h.xreplace(env), {})
            if not isinstance(value, complex):
                usable = False
                break
            if all(decided):
                ours.append(value)
        if not usable:
            continue
        checked += 1
        agreed += int(_same_distinct(ours, wanted))
    return checked, agreed


# --- the operation ---------------------------------------------------------------------------------------------
def _read_equation(
    data: ExpressionInput, limits: Limits, denominators: list[sp.Expr]
) -> tuple[sp.Expr, sp.Expr]:
    sides = split_equation(data.input)
    if sides is None or len(sides) != 2:
        raise ParseError("Write the equation as `left = right` (one equals sign).")
    left, right = (
        parse_expression(
            ExpressionInput(input=side, format=data.format, assumptions=data.assumptions),
            limits,
            denominators,
        )
        for side in sides
    )
    return left, right


def _count(expr: sp.Expr, x: sp.Symbol) -> int:
    return sum(1 for node in sp.preorder_traversal(expr) if node == x)


def _finish(result: IsolateResult, diag: Diagnostics, started: float) -> dict[str, Any]:
    result.diagnostics = diag
    result.duration_ms = int((time.monotonic() - started) * 1000)
    return result.model_dump(mode="json")


def op_isolate(payload: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    limits = Limits(**payload["limits"])
    data = ExpressionInput(**payload["expression"])
    variable = str(payload.get("variable", ""))
    diag = Diagnostics(
        assumptions_declared=len(data.assumptions), collect_detail=bool(payload.get("debug"))
    )
    denominators: list[sp.Expr] = []
    try:
        if not _NAME.match(variable):
            raise ParseError("name the unknown: a letter, then letters, digits or underscores")
        with diag.timed("parse_ms"):
            L, R = _read_equation(data, limits, denominators)
        x = next((s for s in L.free_symbols | R.free_symbols if s.name == variable), None)
        equation_latex = f"{to_latex(L)} = {to_latex(R)}"
        if x is None:
            return _finish(
                IsolateResult(
                    status=Status.PARSE_ERROR,
                    equation_latex=equation_latex,
                    message=f"The unknown {variable!r} does not occur in the equation.",
                ),
                diag,
                started,
            )
        params = sorted((L.free_symbols | R.free_symbols) - {x}, key=lambda s: s.name)
        taken = {s.name for s in params} | {variable}
        rng = random.Random(20260104)  # noqa: S311 - deterministic sampling, not cryptographic
        outside = [d for d in denominators if not d.has(x)]
        global_conds: list[Cond] = []
        for d in outside:
            global_conds.extend(nonzero(d) or [])

        occurrences = _count(L, x) + _count(R, x)
        excluded: list[str] = []
        integers: list[sp.Symbol] = []
        peeled: Peeled | None = None
        solved: Solved | None = None
        with diag.timed("transform_ms"):
            if occurrences == 1:
                G, h = (L, R) if L.has(x) else (R, L)
                peeled = peel(G, h, x, set(taken))
                states, steps, dropped, integers = (
                    peeled.states,
                    peeled.steps,
                    peeled.dropped,
                    peeled.integers,
                )
                method: Literal["inversion", "polynomial"] = "inversion"
            else:
                solved = solve_rational(L, R, x, denominators)
                states, steps, dropped = solved.states, solved.steps, solved.dropped
                global_conds.extend(solved.global_conds)
                excluded = solved.excluded
                method = "polynomial"
        diag.candidates = len(states) + dropped
        if dropped:
            diag.reject(Reason.EXTRANEOUS, dropped)

        states = [State(sp.together(s.h), s.conds) for s in states]
        if len(states) > MAX_SOLUTIONS:
            raise LimitError("too many solutions")

        with diag.timed("verify_ms"):
            checked, agreed, enough = back_substitute(L, R, x, states, params, integers, rng)
            if method == "inversion" and peeled is not None:
                placeholder = sp.Dummy("h")
                G, _ = (L, R) if L.has(x) else (R, L)
                probe = peel(G, placeholder, x, set(taken))
                probes, hits = forward_probe(G, x, placeholder, params, probe, rng)
                complete = probes >= MIN_VALID_POINTS and hits == probes
                note = f"a random value of the unknown was returned by some solution in {hits} of {probes} probes"
            elif solved is not None:
                probes, hits = count_roots(solved, x, params, rng)
                complete = probes >= MIN_VALID_POINTS and hits == probes
                note = f"the distinct roots of the numerator matched the solutions at {hits} of {probes} parameter points"
            else:
                complete, note = False, ""
        verification = Verification(
            symbolic_zero=False,
            numeric_points_checked=checked,
            numeric_points_agreed=agreed,
            passed=bool(states) is False
            or (enough and agreed == checked and checked >= MIN_VALID_POINTS),
            note="each solution substituted back into the original equation at sampled parameter values",
        )
        diag.verification_method = "back_substitution"
        diag.verification_outcome = "passed" if verification.passed and complete else "failed"
        if not (verification.passed and complete):
            diag.reject(Reason.NOT_VERIFIED)
            for state in states:
                diag.note(
                    f"candidate: {x} = {state.h}; conditions: {[c.latex() for c in state.conds]}"
                )
            diag.note(f"substituted back: {agreed}/{checked} agreed; completeness: {note}")
            return _finish(
                IsolateResult(
                    status=Status.NOT_VERIFIED,
                    equation_latex=equation_latex,
                    variable=variable,
                    method=method,
                    verification=verification,
                    reason=Reason.NOT_VERIFIED,
                    message="The candidate solutions did not pass Minerva's checks (substituting back, "
                    "and that none was dropped), so none is shown.",
                ),
                diag,
                started,
            )

        shared = [c for c in _unique(global_conds)]
        if states:
            common = [c for c in states[0].conds if all(c in s.conds for s in states[1:])]
            shared = _unique([*shared, *common])
        everything = [*shared, *(c for s in states for c in s.conds)]
        exceptional: list[str] = []
        for cond in everything:
            line = f"{to_latex(cond.expr)} = 0"
            if cond.kind == "nonzero" and not cond.expr.has(x) and line not in exceptional:
                exceptional.append(line)
        solutions = []
        for state in states:
            own = [c for c in state.conds if c not in shared]
            families = [k for k in integers if state.h.has(k)]
            solutions.append(
                Solution(
                    latex=f"{to_latex(x)} = {to_latex(state.h)}",
                    text=str(state.h),
                    equation_text=f"{x} = {state.h}",
                    conditions_latex=[c.latex() for c in own],
                    branch=", ".join(f"{k} any integer" for k in families),
                    verification=verification,
                )
            )
        assumption_note = (
            f"Solutions are over the complex numbers; the assumptions declared for {variable} "
            "were not used to select among them."
            if data.assumptions.get(variable)
            else ""
        )
        return _finish(
            IsolateResult(
                status=Status.OK,
                equation_latex=equation_latex,
                variable=variable,
                method=method,
                steps=steps,
                solutions=solutions,
                conditions_latex=[c.latex() for c in shared],
                excluded_latex=excluded,
                exceptional=exceptional,
                verification=verification,
                complete=True,
                completeness_note=note,
                assumption_note=assumption_note,
                message=(
                    f"{len(solutions)} solution(s), each substituted back into the equation and checked "
                    "for completeness."
                    if solutions
                    else "The equation has no solution: every candidate was excluded by a condition."
                ),
            ),
            diag,
            started,
        )
    except MinervaMathError as exc:
        status, message = _failure(exc)
        reason = Reason.FORM if isinstance(exc, UnsupportedError) else None
        if reason is not None:
            diag.reject(reason)
        return _finish(IsolateResult(status=status, message=message, reason=reason), diag, started)
