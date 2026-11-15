"""Level families (sums), densities of states (integrals) and separable classical Hamiltonians.

Each builder validates the system, proves what must be proved (convergence, separability) with an
enumerated refusal otherwise, and fills the ``Problem`` with a closed-form single-copy partition
function when one is found. Each system also has a numerical oracle that does not use that route.

Convergence is never assumed. A series is accepted only if the ratio test (applied to the terms and
to E^2 times the terms) gives a limit below 1 for every beta > 0; a series with limit above 1 is
``divergent``; an inconclusive limit is ``convergence_not_provable``. For a density of states the
integrals of E^k rho(E) exp(-beta E), k = 0, 1, 2, must be evaluated by the verified definite
integrator under the declared assumptions. Together with a finite lower energy bound, that justifies
differentiating under the sum or integral (dominated convergence on beta >= beta_0 > 0).
"""

from __future__ import annotations

from typing import Any

import mpmath as mp
import sympy as sp

from ..distributions.models import Condition
from ..integrate import integrate
from ..latexprint import to_latex, to_text
from ..models import ExpressionInput, Status
from .models import ClassicalSystem, DosSystem, FamilySystem, StatStep
from .oracle import BaseOracle, Point, cumulants_from_raw
from .setup import Problem, Refusal

MAX_TERMS = 20000
MAX_FINITE_TERMS = 4000
QUAD_DPS = 30


def _symbols(*exprs: sp.Expr) -> set[sp.Symbol]:
    return {s for e in exprs for s in e.free_symbols if isinstance(s, sp.Symbol)}


def _bound(problem: Problem, text: str, *, allow_infinite: bool) -> sp.Expr:
    t = text.strip()
    if t in {"oo", "+oo", "inf", "\\infty"}:
        if not allow_infinite:
            raise Refusal("unsupported_system", "this bound must be finite")
        return sp.oo
    if t in {"-oo", "-inf", "-\\infty"}:
        if not allow_infinite:
            raise Refusal("unsupported_system", "this bound must be finite")
        return -sp.oo
    return problem.parse(t)


def _with_flag(problem: Problem, name: str, flags: list[str]) -> None:
    problem.assumptions[name] = sorted({*problem.assumptions.get(name, []), *flags})


def definite_integral(
    problem: Problem, integrand: sp.Expr, var: sp.Symbol, lo: sp.Expr, hi: sp.Expr
) -> Any:
    """The verified definite integrator, also for bounds that contain parameters.

    The integrator takes numeric bounds only, so a parametric domain is mapped exactly onto a numeric
    one: x = a + (b - a) u on [0, 1], x = a + u on [0, oo), x = b - u on [0, oo), with the Jacobian
    included. The result is still verified by the integrator's own quadrature at sampled parameters.
    """
    name, bounded = var.name, not (lo.free_symbols or hi.free_symbols)
    if bounded:
        new_var, f, a, b = var, integrand, lo, hi
    else:
        taken = {str(s) for s in integrand.free_symbols | lo.free_symbols | hi.free_symbols}
        u_name = next(f"u{i}" for i in range(100) if f"u{i}" not in taken)
        _with_flag(problem, u_name, ["real", "nonnegative"])
        u = problem.parse_text(u_name)
        if not isinstance(u, sp.Symbol):
            raise Refusal("parse_error", "internal substitution symbol")
        if lo != -sp.oo and hi != sp.oo:
            f = integrand.subs(var, lo + (hi - lo) * u) * (hi - lo)
            a, b = sp.Integer(0), sp.Integer(1)
        elif hi == sp.oo and lo != -sp.oo:
            f, a, b = integrand.subs(var, lo + u), sp.Integer(0), sp.oo
        elif lo == -sp.oo and hi != sp.oo:
            f, a, b = integrand.subs(var, hi - u), sp.Integer(0), sp.oo
        else:
            f, a, b = integrand, lo, hi
        new_var = u if (a, b) != (lo, hi) else var
        name = new_var.name
    return integrate(
        ExpressionInput(input=to_text(f), format="text", assumptions=problem.assumptions),
        name,
        problem.limits,
        lower=to_text(a),
        upper=to_text(b),
    )


# -- level families -----------------------------------------------------------------------
def _ratio_limit(term: sp.Expr, n: sp.Symbol) -> sp.Expr | None:
    if term == 0:
        return None
    try:
        r = sp.simplify(sp.powsimp(term.subs(n, n + 1) / term, force=True))
        return sp.limit(r, n, sp.oo)
    except Exception:
        return None


def _decide_ratio(limit: sp.Expr | None) -> str:
    """'converges', 'diverges' or 'unknown' for the limit of a_{n+1}/a_n."""
    if limit is None or limit.has(sp.nan):
        return "unknown"
    if limit == sp.oo:
        return "diverges"
    if limit == 0:
        return "converges"
    try:
        lg = sp.log(limit)
    except Exception:
        return "unknown"
    if lg.is_negative is True:
        return "converges"
    if lg.is_positive is True:
        return "diverges"
    return "unknown"


def fill_family(problem: Problem, s: FamilySystem) -> set[sp.Symbol]:
    lower_e = _bound(problem, s.lower, allow_infinite=False)
    upper_e = _bound(problem, s.upper, allow_infinite=True)
    if lower_e.is_integer is not True or lower_e.free_symbols:
        raise Refusal("unsupported_system", "the lower index must be an integer constant")
    lo = int(lower_e)
    if upper_e != sp.oo and (upper_e.is_integer is not True or upper_e.free_symbols):
        raise Refusal("unsupported_system", "the upper index must be an integer constant or oo")
    hi = None if upper_e == sp.oo else int(upper_e)
    if hi is not None and hi < lo:
        raise Refusal("parse_error", "the index range is empty")
    _with_flag(problem, s.index, ["integer", *(["nonnegative"] if lo >= 0 else [])])
    n = problem.parse(s.index)
    if not isinstance(n, sp.Symbol):
        raise Refusal("parse_error", "the index must be a plain symbol")
    e = problem.parse(s.energy)
    g = problem.parse(s.degeneracy)
    if e.is_real is not True:
        raise Refusal(
            "assumptions_insufficient",
            "energies must be provably real: declare their symbols real or positive",
        )
    if g.is_integer is not True or g.is_positive is not True:
        raise Refusal(
            "invalid_degeneracy",
            "a degeneracy must be a positive integer for every index in the range",
        )
    if {problem.beta, problem.T, problem.k_B} & (e.free_symbols | g.free_symbols):
        raise Refusal(
            "unsupported_system", "energies and degeneracies must not depend on the temperature"
        )
    beta = problem.beta
    term = g * sp.exp(-beta * e)
    conditions: list[Condition] = []
    if hi is None:
        for label, t in (("a_n", term), ("E_n^2 a_n", e**2 * term)):
            verdict = _decide_ratio(_ratio_limit(t, n))
            if verdict == "diverges":
                raise Refusal(
                    "divergent",
                    f"the series diverges: the ratio of successive terms of {label} tends to a "
                    "limit above 1 (the partition function does not exist for beta > 0)",
                )
            if verdict == "unknown":
                raise Refusal(
                    "convergence_not_provable",
                    f"convergence for every beta > 0 could not be proved: the ratio test on {label} "
                    "is inconclusive",
                )
        conditions.append(
            Condition(
                latex=r"\lim_{n\to\infty} a_{n+1}/a_n < 1\ \ (a_n,\ E_n^2 a_n)",
                established=True,
                how=(
                    "ratio test for every beta > 0: the series and its first two beta-derivatives "
                    "converge, which justifies differentiating term by term"
                ),
            )
        )
    else:
        if hi - lo + 1 > MAX_FINITE_TERMS:
            raise Refusal("resource_limit", f"at most {MAX_FINITE_TERMS} terms")
        conditions.append(
            Condition(
                latex=r"\text{finite sum}",
                established=True,
                how="a finite sum of exponentials always converges",
            )
        )
    z1 = _closed_form_sum(term, n, lo, hi)
    problem.kind = "level_family"
    problem.Z1 = z1
    problem.conditions = conditions
    problem.extra = {"index": n, "energy": e, "degeneracy": g, "lo": lo, "hi": hi, "term": term}
    upper_latex = r"\infty" if hi is None else str(hi)
    problem.setup_steps = [
        StatStep(
            name="levels",
            title="Level family",
            latex=rf"E_{{{to_latex(n)}}} = {to_latex(e)},\quad g_{{{to_latex(n)}}} = {to_latex(g)},\quad "
            rf"{to_latex(n)} = {lo},\dots,{upper_latex}",
        ),
        StatStep(
            name="partition_function",
            title="Partition function",
            latex=rf"Z = \sum_{{{to_latex(n)}={lo}}}^{{{upper_latex}}} g\, e^{{-{to_latex(beta)} E}}"
            + (rf" = {to_latex(z1)}" if z1 is not None else ""),
            text=to_text(z1) if z1 is not None else "",
            result_kind="exact" if z1 is not None else "representation",
            conditions=conditions,
        ),
    ]
    if z1 is None:
        problem.notes.append(
            "No closed form of the sum was found, so it is kept as an unevaluated sum; only numerical "
            "evaluation is available (this does not prove that none exists)."
        )
    return _symbols(e, g) - {n}


def _closed_form_sum(term: sp.Expr, n: sp.Symbol, lo: int, hi: int | None) -> sp.Expr | None:
    if hi is not None:
        if hi - lo + 1 <= 64:
            return sp.Add(*[term.subs(n, k) for k in range(lo, hi + 1)])
        value = sp.summation(term, (n, lo, hi))
    else:
        ratio = sp.simplify(sp.powsimp(term.subs(n, n + 1) / term, force=True))
        if n not in ratio.free_symbols:  # geometric series: a_lo / (1 - r), exact
            return sp.simplify(term.subs(n, lo) / (1 - ratio))
        value = sp.summation(term, (n, lo, sp.oo))
    if value.has(sp.Sum, sp.Piecewise, sp.oo, sp.nan, sp.zoo):
        return None
    return value


class FamilyOracle(BaseOracle):
    """Direct enumeration of the terms with a tail test; no closed form is used."""

    def __init__(self, problem: Problem) -> None:
        super().__init__(problem)
        x = problem.extra
        self._symbols = sorted(
            {*x["energy"].free_symbols, *x["degeneracy"].free_symbols} - {x["index"]},
            key=lambda s: s.name,
        )
        args = [x["index"], *self._symbols]
        self._energy = sp.lambdify(args, x["energy"], modules="mpmath")
        self._degeneracy = sp.lambdify(args, x["degeneracy"], modules="mpmath")
        self.lo: int = x["lo"]
        self.hi: int | None = x["hi"]
        self.tol = mp.mpf(10) ** -15
        self.diff_tol = mp.mpf(10) ** -10

    def raw_moments(self, point: Point, beta: Any) -> list[Any]:
        extra = [point[s] for s in self._symbols]
        sums = [mp.mpf(0)] * 5
        n = self.lo
        quiet = 0
        last = self.hi if self.hi is not None else self.lo + MAX_TERMS
        while n <= last:
            e = self._energy(mp.mpf(n), *extra)
            t = self._degeneracy(mp.mpf(n), *extra) * mp.exp(-beta * e)
            for k in range(5):
                sums[k] += t * e**k
            if self.hi is None and n > self.lo + 8:
                heavy = (1 + abs(e)) ** 4 * t
                quiet = quiet + 1 if heavy < sums[0] * mp.mpf(10) ** -(self.precision - 8) else 0
                if quiet >= 30:
                    return sums
            n += 1
        if self.hi is None:
            raise Refusal(
                "resource_limit",
                f"the series did not settle within {MAX_TERMS} terms at a sampled point",
            )
        return sums

    def lnZ1(self, point: Point, beta: Any) -> Any:
        return mp.log(self.raw_moments(point, beta)[0])

    def cumulants1(self, point: Point, beta: Any, upto: int) -> list[Any]:
        return cumulants_from_raw(self.raw_moments(point, beta), upto)


# -- density of states ---------------------------------------------------------------------
def fill_dos(problem: Problem, s: DosSystem) -> set[sp.Symbol]:
    lo_e = _bound(problem, s.lower, allow_infinite=True)
    hi_e = _bound(problem, s.upper, allow_infinite=True)
    if lo_e == -sp.oo or lo_e == sp.oo:
        raise Refusal(
            "unsupported_system",
            "the density of states needs a finite lower energy bound (energy bounded below)",
        )
    if lo_e.is_real is not True:
        raise Refusal("assumptions_insufficient", "the lower energy bound must be provably real")
    flags = ["real"]
    if lo_e.is_nonnegative is True:
        flags.append("nonnegative")
    _with_flag(problem, s.variable, flags)
    var = problem.parse(s.variable)
    if not isinstance(var, sp.Symbol):
        raise Refusal("parse_error", "the energy variable must be a plain symbol")
    rho = problem.parse(s.density)
    if rho.is_real is not True:
        raise Refusal("assumptions_insufficient", "the density of states must be provably real")
    if {problem.beta, problem.T, problem.k_B} & rho.free_symbols:
        raise Refusal(
            "unsupported_system", "the density of states must not depend on the temperature"
        )
    beta = problem.beta
    if var not in rho.free_symbols:
        problem.notes.append(
            f"The density of states does not depend on {var.name}. In text syntax E is Euler's number: "
            f"write the energy as {var.name} (or choose another variable name)."
        )
    moments: list[sp.Expr] = []
    numeric_only = False
    for k in range(3):
        integrand = var**k * rho * sp.exp(-beta * var)
        res = definite_integral(problem, integrand, var, lo_e, hi_e)
        if res.status is Status.OK and res.result_text:
            moments.append(problem.parse_text(res.result_text))
            continue
        if res.status is Status.CONDITIONS_NOT_SATISFIED:
            raise Refusal(
                "convergence_not_provable",
                "the integral of E^k rho(E) exp(-beta E) converges only under conditions your "
                "assumptions do not imply; declare them and try again",
            )
        if res.status is Status.NOT_VERIFIED:
            raise Refusal(
                "verification_failed",
                "an integral of the density of states could not be verified by quadrature",
            )
        if "diverges" in res.message:
            raise Refusal(
                "convergence_not_provable",
                "SymPy reports that an integral of the density of states diverges; Minerva does not "
                "verify divergence, so it shows no value",
            )
        numeric_only = True
        break
    problem.kind = "density_of_states"
    problem.extra = {"var": var, "rho": rho, "lo": lo_e, "hi": hi_e}
    cond = Condition(
        latex=r"\int E^{k}\rho(E)\,e^{-\beta E}\,dE\ \ (k=0,1,2)\ \text{converge for every }\beta>0",
        established=not numeric_only,
        how=(
            "each was evaluated by the verified definite integrator (quadrature-checked) under the "
            "declared assumptions; with a finite lower bound this justifies differentiating under "
            "the integral (dominated convergence)"
            if not numeric_only
            else "not established symbolically; the numerical quadrature carries an error estimate"
        ),
    )
    problem.conditions = [cond]
    if numeric_only:
        problem.Z1 = None
        problem.notes.append(
            "No closed form of the density-of-states integral was found; only numerical evaluation "
            "is available, and convergence is checked numerically (not proved)."
        )
    else:
        problem.Z1 = moments[0]
        problem.moments1 = moments
    hi_latex = to_latex(hi_e)
    problem.setup_steps = [
        StatStep(
            name="levels",
            title="Density of states",
            latex=rf"\rho({to_latex(var)}) = {to_latex(rho)},\quad {to_latex(lo_e)} \le {to_latex(var)} \le {hi_latex}",
        ),
        StatStep(
            name="partition_function",
            title="Partition function",
            latex=rf"Z = \int \rho({to_latex(var)})\, e^{{-{to_latex(beta)} {to_latex(var)}}}\, d{to_latex(var)}"
            + (rf" = {to_latex(problem.Z1)}" if problem.Z1 is not None else ""),
            text=to_text(problem.Z1) if problem.Z1 is not None else "",
            result_kind="exact" if problem.Z1 is not None else "representation",
            conditions=[cond],
        ),
    ]
    return _symbols(rho, lo_e, hi_e) - {var}


def _quad(f: Any, points: list[Any]) -> Any:
    value, err = mp.quad(f, points, error=True, maxdegree=8)
    if abs(err) > mp.mpf(10) ** -14 * max(abs(value), mp.mpf(10) ** -30):
        raise Refusal(
            "resource_limit",
            "numerical quadrature did not reach its accuracy target at a sample point",
        )
    return value


def _breakpoints(lo: Any, hi: Any) -> list[Any]:
    if mp.isinf(lo) and mp.isinf(hi):
        return [-mp.inf, -10, -1, 0, 1, 10, mp.inf]
    if mp.isinf(hi):
        return [lo, lo + 1, lo + 10, lo + 100, mp.inf]
    if mp.isinf(lo):
        return [-mp.inf, hi - 100, hi - 10, hi - 1, hi]
    return [lo, hi]


class DosOracle(BaseOracle):
    def _state(self, point: Point) -> tuple[Any, ...]:
        return tuple(point[s] for s in self.problem.sample_symbols if s != self.problem.beta)

    precision = QUAD_DPS
    exact = False
    tol = mp.mpf(10) ** -12
    diff_tol = mp.mpf(10) ** -8
    step = mp.mpf(10) ** -5

    def __init__(self, problem: Problem) -> None:
        super().__init__(problem)
        x = problem.extra
        self._symbols = sorted(
            (x["rho"].free_symbols | x["lo"].free_symbols | x["hi"].free_symbols) - {x["var"]},
            key=lambda s: s.name,
        )
        self._rho = sp.lambdify([x["var"], *self._symbols], x["rho"], modules="mpmath")
        self._lo = sp.lambdify(self._symbols, x["lo"], modules="mpmath")
        self._cache: dict[tuple[Any, ...], Any] = {}
        self._hi = (
            (lambda *a: mp.inf)
            if x["hi"] == sp.oo
            else sp.lambdify(self._symbols, x["hi"], modules="mpmath")
        )

    def moment(self, point: Point, beta: Any, k: int) -> Any:
        key = (beta, k, *self._state(point))
        if key not in self._cache:
            extra = [point[s] for s in self._symbols]
            pts = _breakpoints(self._lo(*extra), self._hi(*extra))
            self._cache[key] = _quad(lambda e: e**k * self._rho(e, *extra) * mp.exp(-beta * e), pts)
        return self._cache[key]

    def lnZ1(self, point: Point, beta: Any) -> Any:
        return mp.log(self.moment(point, beta, 0))

    def cumulants1(self, point: Point, beta: Any, upto: int) -> list[Any]:
        return cumulants_from_raw([self.moment(point, beta, k) for k in range(upto + 1)], upto)


# -- classical systems ---------------------------------------------------------------------
def fill_classical(problem: Problem, s: ClassicalSystem) -> set[sp.Symbol]:
    names = [v.name for v in s.variables]
    if len(set(names)) != len(names):
        raise Refusal("parse_error", "phase-space variable names must be distinct")
    for name in names:
        _with_flag(problem, name, ["real"])
    variables = [problem.parse(name) for name in names]
    if not all(isinstance(v, sp.Symbol) for v in variables):
        raise Refusal("parse_error", "phase-space variables must be plain symbols")
    syms: list[sp.Symbol] = [v for v in variables if isinstance(v, sp.Symbol)]
    if s.phase_space_cell is None:
        raise Refusal(
            "convention_unconfirmed",
            "The phase-space normalisation is a convention only you can choose.",
            [
                "What is the phase-space cell, the divisor of the phase-space measure? For example "
                "h^3 for one particle with three degrees of freedom, or 1 for none."
            ],
        )
    cell = problem.parse(s.phase_space_cell)
    if cell.is_positive is not True:
        raise Refusal(
            "assumptions_insufficient",
            "the phase-space cell must be provably positive (declare h positive)",
        )
    h = sp.expand(problem.parse(s.hamiltonian))
    if h.is_real is not True:
        raise Refusal(
            "assumptions_insufficient",
            "the Hamiltonian must be provably real: declare its parameters real",
        )
    if {problem.beta, problem.T, problem.k_B} & h.free_symbols:
        raise Refusal("unsupported_system", "the Hamiltonian must not depend on the temperature")
    groups: dict[sp.Symbol, sp.Expr] = {v: sp.Integer(0) for v in syms}
    constant: sp.Expr = sp.Integer(0)
    for term in sp.Add.make_args(h):
        touched = [v for v in syms if v in term.free_symbols]
        if len(touched) > 1:
            raise Refusal(
                "not_separable",
                f"the Hamiltonian couples {', '.join(v.name for v in touched)} in one term, so the "
                "phase-space integral does not factor into one-variable integrals",
            )
        if touched:
            groups[touched[0]] += term
        else:
            constant += term
    beta = problem.beta
    bounds: dict[sp.Symbol, tuple[sp.Expr, sp.Expr]] = {}
    for v, spec in zip(syms, s.variables, strict=True):
        lo_b = _bound(problem, spec.domain_lower, allow_infinite=True)
        hi_b = _bound(problem, spec.domain_upper, allow_infinite=True)
        for b in (lo_b, hi_b):
            if b.is_real is not True and b not in (sp.oo, -sp.oo):
                raise Refusal("assumptions_insufficient", "domain bounds must be provably real")
        bounds[v] = (lo_b, hi_b)
    factors: list[sp.Expr] = []
    numeric_only = False
    for v in syms:
        integrand = sp.exp(-beta * groups[v])
        res = definite_integral(problem, integrand, v, *bounds[v])
        if res.status is Status.OK and res.result_text:
            factors.append(problem.parse_text(res.result_text))
            continue
        if res.status is Status.CONDITIONS_NOT_SATISFIED:
            raise Refusal(
                "convergence_not_provable",
                f"the integral over {v.name} converges only under conditions your assumptions do "
                "not imply; declare them and try again",
            )
        if res.status is Status.NOT_VERIFIED:
            raise Refusal(
                "verification_failed",
                f"the integral over {v.name} could not be verified by quadrature",
            )
        if "diverges" in res.message:
            raise Refusal(
                "convergence_not_provable",
                f"SymPy reports that the integral over {v.name} diverges (a missing bounding box or "
                "a Boltzmann factor that grows); Minerva does not verify divergence, so it shows no value",
            )
        numeric_only = True
        break
    problem.kind = "classical"
    problem.extra = {
        "variables": syms,
        "groups": groups,
        "constant": constant,
        "cell": cell,
        "domains": [bounds[v] for v in syms],
    }
    cond_sep = Condition(
        latex=r"H=\sum_j h_j(x_j)+c",
        established=True,
        how="the expanded Hamiltonian splits term by term into one-variable parts, so the integral factors exactly",
    )
    cond_int = Condition(
        latex=r"\int e^{-\beta h_j}\,dx_j\ \text{converge}",
        established=not numeric_only,
        how=(
            "each was evaluated by the verified definite integrator (checked by quadrature)"
            if not numeric_only
            else "not established symbolically"
        ),
    )
    problem.conditions = [cond_sep, cond_int]
    if not numeric_only:
        problem.Z1 = sp.exp(-beta * constant) * sp.Mul(*factors) / cell
    problem.notes.append(
        "Phase-space measure: Z_1 = (1/cell) times the integral of exp(-beta H) over the stated domains, "
        "with the cell you chose."
    )
    problem.setup_steps = [
        StatStep(
            name="levels",
            title="Hamiltonian",
            latex=rf"H = {to_latex(h)}",
        ),
        StatStep(
            name="partition_function",
            title="Phase-space partition function",
            latex=rf"Z_1 = \frac{{1}}{{{to_latex(cell)}}}\int e^{{-{to_latex(beta)} H}}\, d\Gamma"
            + (rf" = {to_latex(problem.Z1)}" if problem.Z1 is not None else ""),
            text=to_text(problem.Z1) if problem.Z1 is not None else "",
            result_kind="exact" if problem.Z1 is not None else "representation",
            conditions=[cond_sep, cond_int],
        ),
    ]
    bound_symbols: set[sp.Symbol] = _symbols(
        *(b for pair in bounds.values() for b in pair if b.free_symbols)
    )
    return (_symbols(h, cell) | bound_symbols) - set(syms)


class ClassicalOracle(BaseOracle):
    precision = QUAD_DPS
    exact = False
    tol = mp.mpf(10) ** -12
    diff_tol = mp.mpf(10) ** -8
    step = mp.mpf(10) ** -5

    def __init__(self, problem: Problem) -> None:
        super().__init__(problem)
        x = problem.extra
        self._vars: list[sp.Symbol] = x["variables"]
        self._params = sorted(
            (_symbols(*x["groups"].values(), x["constant"], x["cell"]) - set(self._vars)),
            key=lambda s: s.name,
        )
        self._h = [
            sp.lambdify([v, *self._params], x["groups"][v], modules="mpmath") for v in self._vars
        ]
        self._cache: dict[tuple[Any, ...], list[Any]] = {}
        self._const = sp.lambdify(self._params, x["constant"], modules="mpmath")
        self._cell = sp.lambdify(self._params, x["cell"], modules="mpmath")
        self._domains = [(self._bound_fn(lo), self._bound_fn(hi)) for lo, hi in x["domains"]]

    @staticmethod
    def _bound_fn(expr: sp.Expr) -> Any:
        if expr == sp.oo:
            return lambda point: mp.inf
        if expr == -sp.oo:
            return lambda point: -mp.inf
        syms = sorted(expr.free_symbols, key=lambda s: s.name)
        fn = sp.lambdify(syms, expr, modules="mpmath")
        return lambda point: fn(*(point[s] for s in syms))

    def _state(self, point: Point) -> tuple[Any, ...]:
        """Every sampled value except beta: the cache must not outlive a change of any of them."""
        return tuple(point[s] for s in self.problem.sample_symbols if s != self.problem.beta)

    def _bounds(self, point: Point, j: int) -> tuple[Any, Any]:
        lo, hi = self._domains[j]
        return lo(point), hi(point)

    def factor_moment(self, point: Point, beta: Any, j: int, k: int) -> Any:
        key = (j, k, beta, *self._state(point))
        if key not in self._cache:
            extra = [point[s] for s in self._params]
            lo, hi = self._bounds(point, j)
            pts = _breakpoints(mp.mpmathify(lo), mp.mpmathify(hi))
            h = self._h[j]
            self._cache[key] = _quad(
                lambda x: h(x, *extra) ** k * mp.exp(-beta * h(x, *extra)), pts
            )
        return self._cache[key]

    def lnZ1(self, point: Point, beta: Any) -> Any:
        extra = [point[s] for s in self._params]
        c = self._const(*extra)
        total = -beta * c - mp.log(self._cell(*extra))
        for j in range(len(self._vars)):
            total += mp.log(self.factor_moment(point, beta, j, 0))
        return total

    def cumulants1(self, point: Point, beta: Any, upto: int) -> list[Any]:
        extra = [point[s] for s in self._params]
        total = [mp.mpf(0)] * upto
        for j in range(len(self._vars)):
            raw = [self.factor_moment(point, beta, j, k) for k in range(upto + 1)]
            total = [a + b for a, b in zip(total, cumulants_from_raw(raw, upto), strict=True)]
        total[0] += self._const(*extra)
        return total
