"""Structure of a differential equation, and which approaches can be tried on it.

Everything here is decided by exact tests on the equation (order, linearity, constant coefficients,
whether the highest derivatives can be solved for) or taken from SymPy's `classify_ode` and mapped to a
fixed vocabulary. "Applicable" means that Minerva will try the method on request, never that it will
succeed: each method verifies its own result. Stiffness is not predicted.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import sympy as sp

from ..errors import MinervaMathError
from ..models import Limits, Status
from ..verify import verify_equal
from .models import (
    EquationClass,
    MethodSuggestion,
    OdeAnalysis,
    OdeAnalyzeInput,
    OdeDiagnostics,
    OdeInput,
    OdeInterpretation,
    OdeStage,
    OdeStructure,
    ProblemKind,
)
from .reading import OdeProblem, read_ode

MAX_CLASSIFY_OPS = 300
MAX_EQUILIBRIUM_STATES = 3

_SYMPY_CLASSES: dict[str, EquationClass] = {
    "separable": "separable",
    "1st_linear": "linear_first_order",
    "1st_exact": "exact",
    "Bernoulli": "bernoulli",
    "1st_homogeneous_coeff_best": "homogeneous_degree_zero",
    "1st_homogeneous_coeff_subs_dep_div_indep": "homogeneous_degree_zero",
    "1st_homogeneous_coeff_subs_indep_div_dep": "homogeneous_degree_zero",
    "nth_linear_euler_eq_homogeneous": "euler_cauchy",
    "nth_linear_euler_eq_nonhomogeneous_undetermined_coefficients": "euler_cauchy",
    "nth_linear_euler_eq_nonhomogeneous_variation_of_parameters": "euler_cauchy",
}
_CLASS_TITLES: dict[str, str] = {
    "separable": "separable",
    "linear_first_order": "linear first order",
    "exact": "exact",
    "bernoulli": "Bernoulli",
    "homogeneous_degree_zero": "homogeneous of degree zero",
    "linear_constant_coefficients": "linear with constant coefficients",
    "euler_cauchy": "Euler-Cauchy (equidimensional)",
    "second_order_linear": "second-order linear",
}


def is_zero(expr: sp.Expr) -> bool:
    """True only if `expr` is shown to be zero; an expression too large to simplify is "not shown"."""
    expanded = sp.expand(expr)
    if expanded == 0:
        return True
    if sp.count_ops(expanded) > 200:
        return False
    return bool(sp.simplify(expanded) == 0)


@dataclass
class Solved:
    """The highest derivatives as functions of everything else: `highest[i] = rhs[i]`."""

    highest: list[sp.Symbol]
    rhs: list[sp.Expr]
    restrictions: list[sp.Expr]  # expressions that must be nonzero


def solve_highest(problem: OdeProblem) -> tuple[Solved | None, str]:
    """Solve for the highest derivative of every unknown, when the system is linear in them with an
    invertible coefficient matrix. Returns the reason when it cannot be done: Minerva does not choose
    among the branches of an equation that is nonlinear in its highest derivative."""
    residuals = problem.residuals
    unknowns = problem.unknowns
    if len(residuals) != len(unknowns):
        return None, "the number of equations differs from the number of unknowns"
    if any(problem.orders[u] == 0 for u in unknowns):
        return None, "an unknown is never differentiated: algebraic constraints are not supported"
    highest = [problem.jets[(u, problem.orders[u])] for u in unknowns]
    matrix = sp.Matrix([[f.diff(h) for h in highest] for f in residuals])
    if any(matrix.has(h) for h in highest):
        return None, (
            "the equation is not linear in its highest derivative; Minerva does not choose "
            "among branches"
        )
    rest = sp.Matrix([f.xreplace(dict.fromkeys(highest, 0)) for f in residuals])
    det = matrix.det(method="berkowitz")
    if is_zero(det):
        return None, "the highest derivatives cannot be solved for (singular coefficient matrix)"
    solution = [sp.simplify(x) if sp.count_ops(x) < 150 else x for x in matrix.LUsolve(-rest)]
    for f in residuals:
        check = f.xreplace(dict(zip(highest, solution, strict=True)))
        if not verify_equal(sp.Integer(0), check, symbolic=False).passed:
            return None, "the solved form could not be confirmed numerically"
    restrictions = [] if det.free_symbols == set() else [det]
    return Solved(highest, solution, restrictions), ""


def _linear_parts(
    problem: OdeProblem,
) -> tuple[bool, bool | None, bool | None]:
    """(linear, homogeneous, constant coefficients) in the unknowns and their derivatives."""
    jets = list(problem.jets.values())
    homogeneous = True
    constant = True
    for f in problem.residuals:
        if not f.is_polynomial(*jets) or sp.Poly(f, *jets).total_degree() > 1:
            return False, None, None
        rest = f.xreplace(dict.fromkeys(jets, 0))
        if not is_zero(rest):
            homogeneous = False
        coefficients = [f.diff(j) for j in jets]
        # "Constant coefficients" is about the coefficients of the unknown and its derivatives:
        # a forcing term may depend on the variable.
        if any(c.has(problem.variable) for c in coefficients):
            constant = False
    return True, homogeneous, constant


def _classes(
    problem: OdeProblem, structure_linear: bool, constant: bool | None
) -> list[EquationClass]:
    found: list[EquationClass] = []
    scalar = len(problem.unknowns) == 1 and len(problem.sides) == 1
    order = max(problem.orders.values())
    if scalar and structure_linear:
        f = problem.residuals[0]
        top = f.diff(problem.jets[(problem.unknowns[0], order)])
        if order == 1 and not is_zero(top):
            found.append("linear_first_order")
        if order == 2 and not is_zero(top):
            found.append("second_order_linear")
    if structure_linear and constant:
        found.append("linear_constant_coefficients")
    if scalar and order >= 1:
        f = problem.residuals[0]
        if sp.count_ops(f) <= MAX_CLASSIFY_OPS:
            name = problem.unknowns[0]
            y = sp.Function(name)(problem.variable)
            replace = {
                sym: (sp.Derivative(y, (problem.variable, k)) if k else y)
                for (u, k), sym in problem.jets.items()
                if u == name
            }
            try:
                hints = sp.classify_ode(sp.Eq(f.xreplace(replace), 0), y)
            except (NotImplementedError, ValueError, TypeError, ArithmeticError):
                hints = ()
            for hint in hints:
                mapped = _SYMPY_CLASSES.get(hint)
                if mapped and mapped not in found:
                    found.append(mapped)
    return found


def _same_point(a: sp.Expr, b: sp.Expr) -> bool:
    return bool(a == b or is_zero(a - b))


def _classify_conditions(
    problem: OdeProblem,
) -> tuple[ProblemKind, list[str]]:
    needed = [(u, k) for u in problem.unknowns for k in range(problem.orders[u])]
    total = len(needed)
    given = problem.conditions

    def label(unknown: str, order: int) -> str:
        from .reading import _display

        return f"{_display(unknown, order)}({problem.latex(problem.variable)}_0)"

    if not given:
        return "general", [label(u, k) for u, k in needed]
    points: list[sp.Expr] = []
    for c in given:
        if not any(_same_point(c.at, p) for p in points):
            points.append(c.at)
    have = {(c.unknown, c.order) for c in given}
    if len(points) == 1:
        if len(given) == total and have == set(needed):
            return "initial_value", []
        if len(given) > total or len(have) < len(given):
            return "overdetermined", []
        return "incomplete", [label(u, k) for u, k in needed if (u, k) not in have]
    if len(given) == total:
        return "boundary_value", []
    if len(given) > total:
        return "overdetermined", []
    return "incomplete", []


def suggest_methods(
    problem: OdeProblem,
    structure: OdeStructure,
    classes: list[EquationClass],
    solved: Solved | None,
    solved_reason: str,
    kind: ProblemKind,
) -> list[MethodSuggestion]:
    """Every approach, applicable or not, with the reason. Ranking is by the problem's kind."""
    methods: list[MethodSuggestion] = []
    has_solved = solved is not None
    nonboundary = kind != "boundary_value"
    conditions_text = (
        "initial values for every unknown and derivative below its order"
        if kind in ("general", "incomplete")
        else "the conditions already given"
    )

    symbolic_ok = bool(classes)
    methods.append(
        MethodSuggestion(
            id="symbolic_closed_form",
            title="Closed-form solution (SymPy), verified by substitution",
            applicable=symbolic_ok,
            reason=(
                "recognised as " + ", ".join(_CLASS_TITLES[c] for c in classes)
                if symbolic_ok
                else "no solvable class was recognised, so Minerva does not attempt a closed form"
            ),
            limitations=[
                "SymPy may fail, or give an implicit or unevaluated form",
                "shown only after substituting back into your equation and checking numerically",
                "the solution may omit singular solutions and is valid only where its restrictions hold",
            ],
            tool="ode_solve_symbolic",
            action="closed_form",
        )
    )
    methods.append(
        MethodSuggestion(
            id="numerical_ivp",
            title="Numerical integration (initial-value problem)",
            applicable=has_solved and nonboundary,
            reason=(
                "the highest derivatives can be solved for, so it becomes a first-order system"
                if has_solved and nonboundary
                else (
                    "boundary conditions at two points: use the boundary-value method"
                    if has_solved
                    else solved_reason
                )
            ),
            requires=[
                conditions_text,
                "numerical values for every parameter: ["
                + ", ".join(s.name for s in problem.parameters)
                + "]"
                if problem.parameters
                else "no parameters",
                "the interval of the independent variable",
            ],
            limitations=[
                "an approximation on the interval you choose; not exact and not valid beyond it",
                "stiffness cannot be judged in advance: the integrator is chosen from measured behaviour",
                "checked against an independent integrator, tolerance refinement and the equation's defect",
            ],
            tool="ode_solve_numerical",
            action="ivp",
        )
    )
    methods.append(
        MethodSuggestion(
            id="numerical_bvp",
            title="Numerical boundary-value solution",
            applicable=has_solved and kind == "boundary_value",
            reason=(
                "conditions at two points were given"
                if has_solved and kind == "boundary_value"
                else "needs conditions at two distinct points"
                if has_solved
                else solved_reason
            ),
            requires=[
                "numerical values for every parameter",
                "a starting guess is chosen by Minerva",
            ],
            limitations=[
                "a solution may not exist, may not be unique, and the solver may converge to another one",
                "shown only after the boundary conditions, the residual and a refined mesh agree",
            ],
            tool="ode_solve_numerical",
            action="bvp",
        )
    )
    methods.append(
        MethodSuggestion(
            id="power_series",
            title="Local power series about a regular point",
            applicable=has_solved and nonboundary,
            reason=(
                "the highest derivatives can be solved for, so a Taylor series can be built from the "
                "initial values"
                if has_solved and nonboundary
                else solved_reason or "needs initial values at one point"
            ),
            requires=["the expansion point", conditions_text, "a truncation order"],
            limitations=[
                "local: valid near the expansion point only, and a singular point is unsupported",
                "the residual is checked through the claimed order",
            ],
            tool="ode_solve_symbolic",
            action="series",
        )
    )

    parameters = [s.name for s in problem.parameters]
    methods.append(
        MethodSuggestion(
            id="regular_perturbation",
            title="Regular perturbation expansion",
            applicable=has_solved and nonboundary and bool(parameters),
            reason=(
                "has symbolic parameters, one of which may be small"
                if has_solved and parameters
                else "no symbolic parameter to expand in"
                if has_solved
                else solved_reason
            ),
            requires=["a small parameter named by you (one of: " + ", ".join(parameters) + ")"]
            if parameters
            else ["a small parameter named by you"],
            limitations=[
                "valid for small values of that parameter and bounded intervals",
                "secular terms grow with time and can invalidate the expansion for long times",
            ],
            tool="ode_solve_symbolic",
            action="perturbation",
        )
    )

    pl_reason = _poincare_reason(problem, structure, solved)
    methods.append(
        MethodSuggestion(
            id="poincare_lindstedt",
            title="Poincaré-Lindstedt (periodic motion, one degree of freedom)",
            applicable=pl_reason is None and bool(parameters),
            reason=(
                "a single second-order equation whose acceleration depends only on the position"
                if pl_reason is None and parameters
                else pl_reason or "no symbolic parameter to expand in"
            ),
            requires=["a small parameter named by you", "an equilibrium to oscillate about"],
            limitations=[
                "periodic solutions near a stable equilibrium only",
                "an asymptotic series",
            ],
            tool="ode_solve_symbolic",
            action="perturbation",
        )
    )
    methods.append(
        MethodSuggestion(
            id="multiple_scales",
            title="Multiple scales",
            applicable=False,
            available=False,
            reason="not implemented in Minerva",
        )
    )

    autonomous_states = structure.autonomous and has_solved
    small = structure.state_dimension <= MAX_EQUILIBRIUM_STATES
    methods.append(
        MethodSuggestion(
            id="equilibrium_analysis",
            title="Equilibria and local linear stability",
            applicable=autonomous_states and small,
            reason=(
                "autonomous with few states: equilibria, the Jacobian and its eigenvalues"
                if autonomous_states and small
                else solved_reason
                if not has_solved
                else "the equation depends explicitly on the independent variable"
                if not structure.autonomous
                else f"more than {MAX_EQUILIBRIUM_STATES} states"
            ),
            limitations=[
                "local and from the linearisation only; it says nothing about global stability",
                "where the linearisation has eigenvalues on the imaginary axis it does not decide",
            ],
            tool="ode_analyze",
            action="qualitative",
        )
    )
    methods.append(
        MethodSuggestion(
            id="phase_plane",
            title="Phase plane (direction field and trajectories)",
            applicable=autonomous_states and structure.state_dimension == 2,
            reason=(
                "autonomous with two states"
                if autonomous_states and structure.state_dimension == 2
                else "needs an autonomous equation with exactly two states"
            ),
            limitations=[
                "trajectories are numerical approximations, each checked as in the IVP method"
            ],
            tool="ode_analyze",
            action="qualitative",
        )
    )
    return _rank(methods, kind, symbolic_ok)


def _poincare_reason(
    problem: OdeProblem, structure: OdeStructure, solved: Solved | None
) -> str | None:
    """None when Poincaré-Lindstedt can be tried, else why not."""
    if solved is None:
        return "the equation cannot be solved for its highest derivative"
    if len(problem.unknowns) != 1 or structure.order != 2:
        return "needs a single second-order equation"
    if solved.rhs[0].has(problem.variable):
        return "the acceleration depends explicitly on the independent variable (forcing)"
    if solved.rhs[0].has(problem.jets[(problem.unknowns[0], 1)]):
        return "the acceleration depends on the velocity (damping or a gyroscopic term)"
    return None


def _rank(
    methods: list[MethodSuggestion], kind: ProblemKind, symbolic_ok: bool
) -> list[MethodSuggestion]:
    if kind == "boundary_value":
        order = ["numerical_bvp", "symbolic_closed_form"]
    elif symbolic_ok:
        order = ["symbolic_closed_form", "numerical_ivp", "power_series"]
    else:
        order = ["numerical_ivp", "power_series", "symbolic_closed_form"]
    order += [
        "equilibrium_analysis",
        "phase_plane",
        "poincare_lindstedt",
        "regular_perturbation",
        "numerical_bvp",
        "numerical_ivp",
        "power_series",
        "symbolic_closed_form",
        "multiple_scales",
    ]
    position = {name: i for i, name in enumerate(dict.fromkeys(order))}
    ordered = sorted(methods, key=lambda m: (not m.applicable, position[m.id]))
    rank = 0
    for method in ordered:
        if method.applicable:
            rank += 1
            method.rank = rank
    return ordered


def build_structure(problem: OdeProblem, solved: Solved | None) -> tuple[OdeStructure, bool | None]:
    linear, homogeneous, constant = _linear_parts(problem)
    autonomous = not any(f.has(problem.variable) for f in problem.residuals)
    position_only = (
        solved is not None
        and len(problem.unknowns) == 1
        and problem.orders[problem.unknowns[0]] == 2
        and not solved.rhs[0].has(problem.variable)
        and not solved.rhs[0].has(problem.jets[(problem.unknowns[0], 1)])
    )
    structure = OdeStructure(
        kind="ode" if len(problem.unknowns) == 1 and len(problem.sides) == 1 else "system",
        n_unknowns=len(problem.unknowns),
        n_equations=len(problem.sides),
        orders=dict(problem.orders),
        order=max(problem.orders.values()),
        state_dimension=sum(problem.orders.values()),
        linear=linear,
        homogeneous=homogeneous,
        constant_coefficients=constant,
        autonomous=autonomous,
        solved_for_highest=solved is not None,
        position_only=position_only,
    )
    return structure, constant


def interpretation(problem: OdeProblem) -> OdeInterpretation:
    from .reading import _display

    notes = list(problem.notes)
    notes.append(
        f"{problem.variable.name} is the independent variable, read as real unless you declared otherwise"
    )
    if any(s.name in {"e", "i"} for s in problem.parameters):
        notes.append(
            "e and i are ordinary symbols here; write \\exp(...) or \\exponentialE for Euler's number"
        )
    return OdeInterpretation(
        unknowns=list(problem.unknowns),
        independent=problem.variable.name,
        parameters=[s.name for s in problem.parameters],
        orders=dict(problem.orders),
        equations_latex=[f"{problem.latex(a)} = {problem.latex(b)}" for a, b in problem.sides],
        conditions_latex=[
            f"{_display(c.unknown, c.order)}\\left({problem.latex(c.at)}\\right) = {problem.latex(c.value)}"
            for c in problem.conditions
        ],
        notes=notes,
    )


def _qualitative(
    problem: OdeProblem,
    structure: OdeStructure,
    solved: Solved | None,
    data: OdeAnalyzeInput,
    limits: Limits,
) -> tuple[Any, str | None]:
    """Equilibria and local stability, when the system is autonomous and small; else why not."""
    from .qualitative import MAX_STATES, qualitative

    if solved is None:
        return None, "Equilibria need the system solved for its highest derivatives."
    if not structure.autonomous:
        return (
            None,
            "Equilibria are defined for autonomous systems: this one depends on the variable.",
        )
    if structure.state_dimension > MAX_STATES:
        return None, f"Equilibria are analysed for at most {MAX_STATES} states."
    w = data.window
    window = (w[0], w[1], w[2], w[3]) if w is not None else None
    result = qualitative(
        problem,
        solved.rhs,
        data.parameters,
        window,
        0.4 * limits.timeout_seconds,
    )
    return result, None


def analyze(
    data: OdeInput | OdeAnalyzeInput, limits: Limits, trace_id: str | None = None
) -> OdeAnalysis:
    """Read the equation and describe it. The only failures are an unreadable or unsupported input."""
    stages: list[OdeStage] = []
    started = time.perf_counter()

    def mark(name: str, since: float) -> float:
        now = time.perf_counter()
        stages.append(OdeStage(name=name, ms=round((now - since) * 1000, 2)))
        return now

    diagnostics = OdeDiagnostics(trace_id=trace_id)
    try:
        problem = read_ode(data, limits)
    except MinervaMathError as exc:
        from ..ops import _failure

        status, message = _failure(exc)
        mark("read", started)
        mark("total", started)
        diagnostics.stages = stages
        diagnostics.failure_reason = status.value
        return OdeAnalysis(status=status, message=message, diagnostics=diagnostics)
    clock = mark("read", started)

    solved, reason = solve_highest(problem)
    structure, constant = build_structure(problem, solved)
    clock = mark("structure", clock)

    classes = _classes(problem, structure.linear, constant)
    kind, needed = _classify_conditions(problem)
    clock = mark("classify", clock)

    methods = suggest_methods(problem, structure, classes, solved, reason, kind)
    clock = mark("methods", clock)
    qualitative_result = None
    qualitative_note = None
    if isinstance(data, OdeAnalyzeInput) and data.detail == "qualitative":
        qualitative_result, qualitative_note = _qualitative(
            problem, structure, solved, data, limits
        )
        mark("qualitative", clock)
    mark("total", started)

    notes = [
        "Classes marked by SymPy's classification are candidates: each solver verifies its own result."
    ]
    if qualitative_note:
        notes.append(qualitative_note)
    if solved is None:
        notes.append(f"Not solved for the highest derivative: {reason}.")
    if kind == "overdetermined":
        notes.append("More conditions were given than the order allows.")
    if kind == "incomplete" and problem.conditions:
        notes.append("Fewer conditions were given than the order requires.")
    diagnostics.stages = stages
    diagnostics.n_equations = len(problem.sides)
    diagnostics.order = structure.order
    diagnostics.state_dimension = structure.state_dimension
    diagnostics.classes = list(classes)

    solved_latex: list[str] = []
    restrictions: list[str] = []
    if solved is not None:
        for h, rhs in zip(solved.highest, solved.rhs, strict=True):
            solved_latex.append(f"{problem.latex(h)} = {problem.latex(rhs)}")
        restrictions = [f"{problem.latex(r)} \\neq 0" for r in solved.restrictions]

    return OdeAnalysis(
        status=Status.OK,
        interpretation=interpretation(problem),
        structure=structure,
        problem_kind=kind,
        conditions_needed=needed,
        classes=classes,
        solved_latex=solved_latex,
        restrictions_latex=restrictions,
        methods=methods,
        qualitative=qualitative_result,
        notes=notes,
        diagnostics=diagnostics,
    )
