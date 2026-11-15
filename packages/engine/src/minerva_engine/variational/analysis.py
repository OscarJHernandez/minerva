"""Structure of a functional: orders, regularity, boundary slots, and the suggested next steps."""

from __future__ import annotations

import sympy as sp

from .jets import latex_jet_name
from .models import (
    Analysis,
    ConditionState,
    ConditionStatus,
    End,
    FunctionInfo,
    Need,
    NextStep,
    Regularity,
)
from .setup import Problem

ENDS: tuple[End, ...] = ("lower", "upper")


def legendre_matrix(problem: Problem) -> tuple[sp.Matrix, list[int]]:
    """W_ij = ∂²L / ∂u_i^(n_i) ∂u_j^(n_j) over the functions that appear in the integrand.

    `n_i` is the highest derivative of function i *in the integrand*, so mixed orders (a
    Hamiltonian-style `p ẋ - H(x, p)` has orders (1, 0)) are handled. Returns the matrix and the
    function indices it is built over.
    """
    jets = problem.jets
    rows = [i for i in range(jets.count) if jets.order_in(problem.lagrangian, i) >= 0]
    top = {i: jets.u[i][jets.orders[i]] for i in rows}
    matrix = (
        sp.Matrix([[sp.diff(problem.lagrangian, top[i], top[j]) for j in rows] for i in rows])
        if rows
        else sp.zeros(0, 0)
    )
    return matrix, rows


def regularity_of(problem: Problem) -> tuple[Regularity, sp.Expr]:
    matrix, rows = legendre_matrix(problem)
    if not rows:
        return "degenerate", sp.Integer(0)
    det = sp.factor(sp.expand(matrix.det(method="berkowitz")))
    if det == 0:
        return "degenerate", det
    if det.is_number and det != 0:
        return "regular", det
    if det.is_nonzero is True:
        return "regular", det
    return "conditional", det


def boundary_slots(problem: Problem) -> list[tuple[End, int, int]]:
    """(end, function, order j < n_i): the derivatives whose perturbation multiplies a boundary term."""
    jets = problem.jets
    return [(end, i, j) for end in ENDS for i in range(jets.count) for j in range(jets.orders[i])]


def condition_state(problem: Problem, end: End, i: int, j: int) -> ConditionState:
    cond = problem.conditions.get((end, i, j))
    return "unspecified" if cond is None else cond.kind


def analyze(problem: Problem) -> Analysis:
    jets = problem.jets
    L = problem.lagrangian
    regularity, det = regularity_of(problem)

    functions = [
        FunctionInfo(
            name=name,
            name_latex=latex_jet_name(name, 0),
            order=jets.orders[i],
            appears=jets.order_in(L, i) >= 0,
        )
        for i, name in enumerate(jets.names)
    ]

    statuses: list[ConditionStatus] = []
    listed = {(end, i, j) for end, i, j in boundary_slots(problem)}
    for end, i, j in [*boundary_slots(problem), *sorted(set(problem.conditions) - listed)]:
        state = condition_state(problem, end, i, j)
        cond = problem.conditions.get((end, i, j))
        statuses.append(
            ConditionStatus(
                function=jets.names[i],
                order=j,
                end=end,
                state=state,
                value_latex=None if cond is None or cond.value is None else cond.value,
            )
        )
    unspecified = sum(1 for s in statuses if s.state == "unspecified")

    explicit = problem.variable in L.free_symbols
    interval_known = problem.bounds_numeric
    needs_values = bool(problem.parameters)
    lo, hi = problem.latex(problem.lower), problem.latex(problem.upper)
    names = ", ".join(jets.names)
    parts = [
        rf"S[{names}]=\int_{{{lo}}}^{{{hi}}}\left({problem.latex(L)}\right)\mathrm{{d}}{problem.variable.name}"
    ]
    if problem.upper_term != 0:
        parts.append(
            rf"+\left.\left({problem.latex(problem.upper_term)}\right)\right|_{{{problem.variable.name}={hi}}}"
        )
    if problem.lower_term != 0:
        parts.append(
            rf"+\left.\left({problem.latex(problem.lower_term)}\right)\right|_{{{problem.variable.name}={lo}}}"
        )

    steps: list[NextStep] = [
        NextStep(id="first_variation"),
        NextStep(id="euler_lagrange"),
    ]
    if boundary_slots(problem) or problem.has_endpoint_terms:
        steps.append(NextStep(id="boundary_terms"))
    if unspecified:
        steps.append(NextStep(id="declare_conditions", needs=["endpoint_conditions"]))
    numeric_needs: list[Need] = []
    if not interval_known:
        numeric_needs.append("interval")
    if needs_values:
        numeric_needs.append("parameter_values")
    steps.append(NextStep(id="evaluate_action", needs=[*numeric_needs, "trial_path"]))
    steps.append(NextStep(id="verify_variation", needs=[*numeric_needs, "trial_path"]))
    solve_needs = [*numeric_needs]
    if unspecified or not problem.conditions:
        solve_needs.append("endpoint_conditions")
    steps.append(NextStep(id="solve_stationary", needs=[*solve_needs, "initial_guess"]))
    steps.append(NextStep(id="second_variation", needs=[*numeric_needs, "trial_path"]))

    return Analysis(
        functional_latex="".join(parts),
        variable=problem.variable.name,
        lower_latex=lo,
        upper_latex=hi,
        functions=functions,
        parameters=[p.name for p in problem.parameters],
        given_functions=sorted(problem.given),
        highest_order=max(jets.orders),
        explicit_in_variable=explicit,
        has_endpoint_terms=problem.has_endpoint_terms,
        regularity=regularity,
        hessian_det_latex=problem.latex(det),
        conditions=statuses,
        unspecified=unspecified,
        conditions_needed_per_end=sum(
            jets.orders[i] for i in range(jets.count) if jets.order_in(L, i) >= 0
        ),
        reading=problem.reading,
        next_steps=steps,
    )
