"""The `solve` action: stationary paths by two methods, and the gates that decide what is shown.

* Method B (`hermite.py`) discretises the action and solves `∇S_h = 0` on three meshes.
* Method A (`cheb.py`) solves the derived Euler-Lagrange equations and natural boundary conditions
  by Chebyshev collocation on three node counts (at most two unknown functions).
* A path is shown only if its method's refinement sequence converged *and* the method-independent
  stationarity check passes: the first variation of the original integrand, along several
  admissible perturbations, vanishes relative to the size of its terms.

Wording is deliberate: a stationary path found from this starting guess; it may be a minimum, a
maximum or a saddle; it is not claimed to be unique.
"""

from __future__ import annotations

from typing import Any

import mpmath as mp

from ..errors import ParseError, UnsupportedInput
from ..models import Status
from . import cheb, hermite
from .action import action_of
from .firstvar import FirstVariationData, first_variation
from .models import (
    Diagnostics,
    End,
    MethodReport,
    PathSamples,
    Solution,
    StationarityCheck,
    StationaryPath,
    VariationInput,
    VariationReport,
)
from .numeric_setup import MissingValues, path_field, prepare
from .numerics import JetField, Numerics, _to_mp
from .paths import numeric_value
from .secondrun import diagnostics
from .secondvar import second_form
from .setup import Problem
from .userverify import _direct_claim, _poly

HERMITE_LEVELS = (16, 32, 64)
CHEB_LEVELS = (20, 28, 36)
STATIONARY_TOL = mp.mpf("1e-6")
AGREE_TOL = 1e-6
SAMPLES = 41


class LineGuess:
    """Straight line between the fixed end values (a free end copies the other end; none: zero)."""

    def __init__(self, num: Numerics, fixed: dict[tuple[End, int, int], Any]) -> None:
        self.num = num
        jets = num.jets
        self.values = []
        for i in range(jets.count):
            lo, hi = fixed.get(("lower", i, 0)), fixed.get(("upper", i, 0))
            lo = lo if lo is not None else (hi if hi is not None else mp.mpf(0))
            hi = hi if hi is not None else lo
            self.values.append((lo, hi))

    def at(self, s: Any) -> list[list[Any]]:
        a, b = self.num.lower, self.num.upper
        out = []
        for i, (lo, hi) in enumerate(self.values):
            n = 2 * self.num.jets.orders[i] + 1
            row = [lo + (hi - lo) * (s - a) / (b - a), (hi - lo) / (b - a)] + [mp.mpf(0)] * (n - 1)
            out.append(row[: n + 1])
        return out


def _validate(problem: Problem) -> None:
    jets = problem.jets
    for i, n in enumerate(jets.orders):
        if jets.order_in(problem.lagrangian, i) < 0:
            raise UnsupportedInput(
                f"{jets.names[i]} does not appear in the integrand: nothing determines it"
            )
        if n < 1:
            raise UnsupportedInput(
                f"{jets.names[i]} appears without a derivative (an algebraic unknown): "
                "eliminate it before solving"
            )
        if n > 2:
            raise UnsupportedInput("solving is supported for derivatives up to order 2")
    if sum(jets.orders) > 4:
        raise UnsupportedInput(
            "solving is supported when the orders of the functions sum to at most 4"
        )
    missing: list[str] = []
    for end in ("lower", "upper"):
        for i, n in enumerate(jets.orders):
            for j in range(n):
                cond = problem.conditions.get((end, i, j))
                if cond is None:
                    missing.append(f"{jets.names[i]}^({j}) at the {end} end")
                elif cond.kind == "fixed" and cond.value is None:
                    missing.append(f"the value of {jets.names[i]}^({j}) at the {end} end")
    if missing:
        raise MissingValues(
            "Declare each endpoint condition as fixed (with a value) or free: " + ", ".join(missing)
        )
    for end, i, j in problem.conditions:
        if j >= jets.orders[i]:
            raise MissingValues(
                f"{jets.names[i]}^({j}) at the {end} end is an extra condition: the variation "
                "produces conditions only on derivatives below the order of the integrand, "
                "so the problem would be over-determined"
            )
    for term in (problem.lower_term, problem.upper_term):
        for i, k in jets.jets_in(term):
            if k >= jets.orders[i]:
                raise MissingValues(
                    "an endpoint term depends on a derivative of the same order as the integrand's: "
                    "that adds a condition beyond the Euler-Lagrange problem"
                )


def _sample_points(num: Numerics) -> list[Any]:
    return [num.lower + (num.upper - num.lower) * k / (SAMPLES - 1) for k in range(SAMPLES)]


def _distance(num: Numerics, a: JetField, b: JetField) -> float:
    worst = mp.mpf(0)
    scale = mp.mpf(1)
    for s in _sample_points(num):
        ra, rb = a.at(s), b.at(s)
        for i in range(len(ra)):
            worst = max(worst, abs(ra[i][0] - rb[i][0]))
            scale = max(scale, abs(rb[i][0]))
    return float(worst / scale)


def _convergence(differences: list[float], tolerance: float) -> bool:
    if len(differences) < 2:
        return False
    first, last = differences[-2], differences[-1]
    return last <= tolerance and (last <= first / 3 or last <= tolerance * 1e-3)


def _run_hermite(
    num: Numerics, fixed: dict[tuple[End, int, int], Any], guess: JetField
) -> tuple[MethodReport, JetField | None]:
    report = MethodReport(method="hermite_action", ran=True, converged=False)
    disc = hermite.discretise(num)
    fields: list[JetField] = []
    start: JetField = guess
    try:
        for elements in HERMITE_LEVELS:
            mesh = hermite.Mesh(disc, elements)
            boundary = hermite.boundary_of(mesh, {k: float(v) for k, v in fixed.items()})
            z, iterations = hermite.newton(mesh, hermite.nodal_guess(mesh, start, 0), boundary)
            field = hermite.HermiteField(mesh, z)
            fields.append(field)
            start = field
            report.levels.append(elements)
            report.newton_iterations.append(iterations)
    except hermite.NotConverged as exc:
        report.note = f"did not converge from this starting guess: {exc}"
        return report, None
    report.successive_differences = [
        _distance(num, fields[k], fields[k + 1]) for k in range(len(fields) - 1)
    ]
    report.converged = _convergence(report.successive_differences, 1e-7)
    if not report.converged:
        report.note = "the refinement sequence did not settle"
    return report, fields[-1]


def _run_chebyshev(
    num: Numerics,
    derived: FirstVariationData,
    fixed: dict[tuple[End, int, int], Any],
    guess: JetField,
) -> tuple[MethodReport, JetField | None]:
    report = MethodReport(method="chebyshev_euler_lagrange", ran=True, converged=False)
    if num.jets.count > 2:
        report.ran = False
        report.note = "not run: collocation is limited to two unknown functions"
        return report, None
    fields: list[JetField] = []
    start: JetField = guess
    try:
        for nodes in CHEB_LEVELS:
            field = cheb.solve_chebyshev(num, derived, fixed, start, nodes)
            fields.append(field)
            start = field
            report.levels.append(nodes)
            report.newton_iterations.append(0)
    except hermite.NotConverged as exc:
        report.note = f"did not converge from this starting guess: {exc}"
        return report, None
    report.successive_differences = [
        _distance(num, fields[k], fields[k + 1]) for k in range(len(fields) - 1)
    ]
    report.converged = _convergence(report.successive_differences, 1e-9)
    if not report.converged:
        report.note = "the refinement sequence did not settle"
    return report, fields[-1]


def _stationary(
    num: Numerics, problem: Problem, derived: FirstVariationData, field: JetField
) -> StationarityCheck:
    claim = _direct_claim(num, derived, None)
    degree = max(2 * max(num.jets.orders) + 3, len(problem.conditions) + 4)
    worst = mp.mpf(0)
    floor = mp.mpf("1e-12") * (1 + abs(action_of(num, field)[0]))
    trials = 0
    for kind in ("bubble", "free_end", "free_end"):
        eta = num.field(_poly(num, kind, None, degree))
        value, scale, _ = claim(field, eta)
        worst = max(worst, abs(value) / (scale + floor))
        trials += 1
    return StationarityCheck(
        residual=float(worst),
        trials=trials,
        tolerance=float(STATIONARY_TOL),
        verdict="stationary_within_tolerance" if worst <= STATIONARY_TOL else "not_stationary",
    )


def _samples(num: Numerics, field: JetField) -> PathSamples:
    ts = _sample_points(num)
    names = num.jets.names
    values: dict[str, list[float]] = {n: [] for n in names}
    slopes: dict[str, list[float]] = {n: [] for n in names}
    for s in ts:
        rows = field.at(s)
        for i, n in enumerate(names):
            values[n].append(float(mp.re(rows[i][0])))
            slopes[n].append(float(mp.re(rows[i][1])) if len(rows[i]) > 1 else 0.0)
    return PathSamples(t=[float(s) for s in ts], values=values, slopes=slopes)


def solve(problem: Problem, data: VariationInput, deadline: float) -> VariationReport:
    _validate(problem)
    num = prepare(problem, data, seed=20260601)
    if any(abs(mp.im(v)) != 0 for v in num.params.values()):
        raise MissingValues("Stationary paths need real parameter values.")
    fixed: dict[tuple[End, int, int], Any] = {
        key: mp.re(_to_mp(numeric_value(problem, cond.value or "0", data.format)))
        for key, cond in problem.conditions.items()
        if cond.kind == "fixed"
    }
    derived = first_variation(problem)
    if len(data.paths) > 1:
        raise ParseError("give at most one path, used as the starting guess")
    guess: JetField
    if data.paths:
        guess, _ = path_field(
            num, problem, data.paths[0], data, depth=2 * max(problem.jets.orders) + 1
        )
    else:
        guess = LineGuess(num, fixed)

    hermite_report, hermite_field = _run_hermite(num, fixed, guess)
    cheb_report, cheb_field = _run_chebyshev(num, derived, fixed, guess)
    if (
        not cheb_report.converged
        and cheb_report.ran
        and hermite_field is not None
        and hermite_report.converged
    ):
        retry_report, retry_field = _run_chebyshev(num, derived, fixed, hermite_field)
        if retry_report.converged:
            retry_report.note = "converged only when started from the other method's path"
            cheb_report, cheb_field = retry_report, retry_field

    candidates: list[tuple[str, JetField]] = []
    if hermite_report.converged and hermite_field is not None:
        candidates.append(("hermite_action", hermite_field))
    if cheb_report.converged and cheb_field is not None:
        candidates.append(("chebyshev_euler_lagrange", cheb_field))
    methods = [hermite_report, cheb_report]
    if not candidates:
        return VariationReport(
            status=Status.NOT_VERIFIED,
            action="solve",
            solution=Solution(paths=[], methods=methods),
            message="No stationary path was confirmed from this starting guess. "
            "Try another starting path; nothing is claimed about whether one exists.",
        )

    paths: list[StationaryPath] = []
    kept: list[tuple[str, JetField]] = []
    notes: list[str] = []
    for name, field in candidates:
        check = _stationary(num, problem, derived, field)
        if check.verdict != "stationary_within_tolerance":
            notes.append(f"the {name} path failed the method-independent stationarity check")
            continue
        total, _, integral = action_of(num, field)
        if integral.error > mp.mpf("1e-8") * max(1, abs(total)):
            notes.append(f"the quadrature rules disagreed for the {name} path")
            continue
        kept.append((name, field))
        paths.append(
            StationaryPath(
                method=name,
                samples=_samples(num, field),
                action=str(mp.nstr(total, 15)),
                action_quadrature_error=float(integral.error),
                stationarity=check,
            )
        )
    if not paths:
        return VariationReport(
            status=Status.NOT_VERIFIED,
            action="solve",
            solution=Solution(paths=[], methods=methods, notes=notes),
            message="The computed paths did not pass the stationarity check, so none is shown.",
        )

    agreement: float | None = None
    distinct = False
    confirmed = False
    if len(kept) == 2:
        agreement = _distance(num, kept[0][1], kept[1][1])
        if agreement <= AGREE_TOL:
            confirmed = True
            paths = [paths[1]]  # the collocation path is the more accurate one
            kept = [kept[1]]
        else:
            distinct = True
            notes.append(
                "The two methods ended at different stationary paths from the same starting "
                "guess; each passed its own refinement and stationarity checks."
            )
    else:
        notes.append("Only one method confirmed a path, so it was not independently cross-checked.")

    second = None
    if data.with_second:
        second = diagnostics(problem, num, kept[0][1], second_form(problem))

    return VariationReport(
        status=Status.OK,
        action="solve",
        solution=Solution(
            paths=paths,
            methods=methods,
            agreement=agreement,
            confirmed_by_both=confirmed,
            distinct_paths=distinct,
            second=second,
            notes=notes,
        ),
        message="A stationary path was found from this starting guess.",
        diagnostics=Diagnostics(stages=len(paths)),
    )
