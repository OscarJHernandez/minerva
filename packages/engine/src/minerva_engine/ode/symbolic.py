"""Closed-form solutions of differential equations through SymPy's `dsolve`, shown only when verified.

SymPy proposes; `verification.py` decides. A candidate is accepted only if the equation (as the user
wrote it) holds when the candidate is substituted, with derivatives taken numerically, at enough
random points. Initial conditions are applied afterwards and checked separately. "SymPy found no
closed form" is never reported as "none exists", singular solutions are not searched, and a family is
called general only when its constants are numerically independent and as many as the order.
"""

from __future__ import annotations

import contextlib
import re
import signal
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal

import sympy as sp

from ..errors import MinervaMathError
from ..latexprint import to_latex
from ..models import Limits, Status
from ..restrictions import domain_restrictions
from .analyze import interpretation
from .models import (
    ConditionCheck,
    OdeDiagnostics,
    OdeInput,
    OdeSolution,
    OdeSolutionItem,
    OdeStage,
    OdeVerification,
)
from .reading import OdeProblem, _latex_name, read_ode
from .verification import (
    TOLERANCE,
    Residual,
    condition_residual,
    constants_of,
    implicit_check,
    independent_constants,
    residual_check,
    symbolic_residual_is_zero,
)

MAX_RESULT_OPS = 3000
MAX_CANDIDATES = 6


class SolverTimeout(Exception):
    """SymPy did not finish inside the budget."""


@contextlib.contextmanager
def deadline(seconds: float) -> Iterator[None]:
    """Interrupt the block after `seconds` (Unix, main thread). Elsewhere the sandbox's wall-clock limit
    still applies, so this only makes the failure explicit and keeps the analysis available."""
    if threading.current_thread() is not threading.main_thread() or not hasattr(
        signal, "setitimer"
    ):
        yield
        return

    def expire(signum: int, frame: object) -> None:
        raise SolverTimeout

    previous = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, max(0.05, seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@dataclass
class Candidate:
    kind: Literal["explicit", "implicit", "unevaluated"]
    outputs: list[sp.Expr] = field(default_factory=list)  # one per unknown (explicit, unevaluated)
    relation: sp.Expr | None = None  # G(x, y; C) = 0 (implicit)

    @property
    def expressions(self) -> list[sp.Expr]:
        return [self.relation] if self.relation is not None else self.outputs


def _function_form(
    problem: OdeProblem,
) -> tuple[list[sp.Eq], list[sp.Expr], dict[sp.Symbol, sp.Expr]]:
    functions = {u: sp.Function(u)(problem.variable) for u in problem.unknowns}
    replace: dict[sp.Symbol, sp.Expr] = {}
    for (unknown, order), jet in problem.jets.items():
        replace[jet] = (
            sp.Derivative(functions[unknown], (problem.variable, order))
            if order
            else functions[unknown]
        )
    equations = [sp.Eq(f.xreplace(replace), 0) for f in problem.residuals]
    return equations, [functions[u] for u in problem.unknowns], replace


def _equations_in(raw: object) -> list[list[sp.Eq]]:
    """dsolve returns one Eq, a list of Eq (several solutions, or one per unknown of a system), or a
    list of such lists."""
    if isinstance(raw, sp.Eq):
        return [[raw]]
    if isinstance(raw, list | tuple):
        if all(isinstance(item, sp.Eq) for item in raw):
            return [list(raw)]
        if all(isinstance(item, list | tuple) for item in raw):
            return [list(item) for item in raw]
    return []


def _candidates(raw: object, problem: OdeProblem, functions: list[sp.Expr]) -> list[Candidate]:
    result: list[Candidate] = []
    scalar = len(functions) == 1
    to_jets = {f: problem.jets[(u, 0)] for f, u in zip(functions, problem.unknowns, strict=True)}
    groups = _equations_in(raw)
    if scalar and groups and len(groups) == 1:
        groups = [[eq] for eq in groups[0]]  # a list of Eq is a list of solutions
    for group in groups:
        by_function = {}
        for eq in group:
            if eq.lhs in functions:
                by_function[eq.lhs] = eq
        if scalar:
            eq = group[0]
            if eq.lhs == functions[0] and not eq.rhs.has(functions[0]):
                outputs = [eq.rhs]
                result.append(
                    Candidate(
                        "unevaluated" if eq.rhs.has(sp.Integral) else "explicit", outputs=outputs
                    )
                )
            else:
                relation = (eq.lhs - eq.rhs).xreplace(to_jets)
                result.append(Candidate("implicit", relation=relation))
        elif len(by_function) == len(functions) and all(
            not by_function[f].rhs.has(*functions) for f in functions
        ):
            outputs = [by_function[f].rhs for f in functions]
            result.append(
                Candidate(
                    "unevaluated" if any(o.has(sp.Integral) for o in outputs) else "explicit",
                    outputs=outputs,
                )
            )
    return result


def _name_latex(problem: OdeProblem, unknown: str) -> str:
    return f"{_latex_name(unknown)}\\left({to_latex(problem.variable)}\\right)"


def _show(problem: OdeProblem, cand: Candidate) -> list[str]:
    if cand.relation is not None:
        return [f"{problem.latex(cand.relation)} = 0"]
    return [
        f"{_name_latex(problem, u)} = {to_latex(o)}"
        for u, o in zip(problem.unknowns, cand.outputs, strict=True)
    ]


def _restrictions(cand: Candidate) -> list[str]:
    notes: list[str] = []
    for expr in cand.expressions:
        found, _ = domain_restrictions(sp.Integer(1), expr)
        for text in found:
            if text not in notes:
                notes.append(text)
    return notes


def _check(
    problem: OdeProblem, cand: Candidate, constants: list[sp.Symbol], *, symbolic: bool
) -> tuple[OdeVerification, Residual]:
    if cand.relation is not None:
        residual = implicit_check(problem, cand.relation, constants)
        return residual.verification(), residual
    unevaluated = cand.kind == "unevaluated"
    residual = residual_check(problem, cand.outputs, constants, unevaluated=unevaluated)
    zero = (
        symbolic_residual_is_zero(problem, cand.outputs) if symbolic and not unevaluated else None
    )
    rank = (
        independent_constants(problem, cand.outputs, constants, unevaluated=unevaluated)
        if constants
        else 0
    )
    return residual.verification(symbolic_zero=zero, independent_constants=rank), residual


def _family(
    problem: OdeProblem, constants: list[sp.Symbol], rank: int | None, fixed: bool
) -> Literal["general", "particular", "partial"]:
    dimension = sum(problem.orders.values())
    if not constants:
        return "particular"
    if fixed:
        return "partial"
    return "general" if rank == dimension and len(constants) == dimension else "partial"


def _implicit_depends_on_constant(problem: OdeProblem, relation: sp.Expr, c: sp.Symbol) -> bool:
    """For a first-order implicit family with one constant: the relation really changes with it."""
    return not sp.simplify(relation.diff(c)).is_zero


def _fix_constants(
    problem: OdeProblem, cand: Candidate, constants: list[sp.Symbol]
) -> tuple[list[Candidate], str]:
    """Candidates with the constants chosen so that the conditions hold, and a note if none do."""
    equations: list[sp.Expr] = []
    y_of = {u: problem.jets[(u, 0)] for u in problem.unknowns}
    for c in problem.conditions:
        if cand.relation is not None:
            if c.order:
                return [], "conditions on derivatives are not applied to implicit solutions"
            relation = cand.relation.xreplace({y_of[c.unknown]: c.value})
            equations.append(relation.xreplace({problem.variable: c.at}))
        else:
            index = problem.unknowns.index(c.unknown)
            expr = cand.outputs[index]
            expr = sp.diff(expr, problem.variable, c.order) if c.order else expr
            equations.append(expr.xreplace({problem.variable: c.at}) - c.value)
    if any(e.has(sp.zoo, sp.nan, sp.oo, -sp.oo) for e in equations):
        return [], "the conditions fall where the solution is not defined"
    if not constants:
        return [], "there are no constants to fit the conditions"
    try:
        solutions = sp.solve(equations, constants, dict=True)
    except (NotImplementedError, ValueError, TypeError, RecursionError):
        return [], "the constants could not be solved from the conditions"
    if not solutions:
        return [], "no member of the family found satisfies the conditions"
    fitted: list[Candidate] = []
    for solution in solutions[:MAX_CANDIDATES]:
        if cand.relation is not None:
            fitted.append(Candidate("implicit", relation=cand.relation.xreplace(solution)))
        else:
            fitted.append(
                Candidate(cand.kind, outputs=[o.xreplace(solution) for o in cand.outputs])
            )
    return fitted, ""


def _condition_checks(
    problem: OdeProblem, cand: Candidate, constants: list[sp.Symbol]
) -> list[ConditionCheck]:
    from .reading import _display

    checks = []
    for c in problem.conditions:
        label = (
            f"{_display(c.unknown, c.order)}\\left({problem.latex(c.at)}\\right) = "
            f"{problem.latex(c.value)}"
        )
        if cand.relation is not None:
            residual: float | None = None
            y = problem.jets[(c.unknown, 0)]
            value = cand.relation.xreplace({y: c.value, problem.variable: c.at})
            try:
                residual = abs(float(sp.N(value, 30))) if not value.free_symbols else None
            except (TypeError, ValueError):
                residual = None
        else:
            residual = condition_residual(
                problem,
                cand.outputs,
                constants,
                c.unknown,
                c.order,
                c.at,
                c.value,
            )
        checks.append(
            ConditionCheck(
                latex=label,
                satisfied=residual is not None and residual <= TOLERANCE,
                residual=residual,
            )
        )
    return checks


def solve_closed_form(data: OdeInput, limits: Limits, trace_id: str | None = None) -> OdeSolution:
    started = time.perf_counter()
    stages: list[OdeStage] = []
    diagnostics = OdeDiagnostics(trace_id=trace_id, method="closed_form")

    def mark(name: str, since: float) -> float:
        now = time.perf_counter()
        stages.append(OdeStage(name=name, ms=round((now - since) * 1000, 2)))
        return now

    def finish(solution: OdeSolution) -> OdeSolution:
        mark("total", started)
        diagnostics.stages = stages
        solution.diagnostics = diagnostics
        return solution

    try:
        problem = read_ode(data, limits)
    except MinervaMathError as exc:
        from ..ops import _failure

        status, message = _failure(exc)
        diagnostics.failure_reason = status.value
        return finish(OdeSolution(status=status, message=message))
    clock = mark("read", started)
    meaning = interpretation(problem)
    diagnostics.n_equations = len(problem.sides)
    diagnostics.order = max(problem.orders.values())
    diagnostics.state_dimension = sum(problem.orders.values())

    def refuse(status: Status, message: str, reason: str) -> OdeSolution:
        diagnostics.failure_reason = reason
        return finish(OdeSolution(status=status, message=message, interpretation=meaning))

    if len(problem.sides) != len(problem.unknowns):
        return refuse(
            Status.UNSUPPORTED,
            "the number of equations differs from the number of unknowns",
            "equation_count",
        )
    if any(order == 0 for order in problem.orders.values()):
        return refuse(
            Status.UNSUPPORTED,
            "an unknown is never differentiated: algebraic constraints are not supported",
            "algebraic",
        )
    if any(re.fullmatch(r"C_?\{?\d+\}?", p.name) for p in problem.parameters):
        return refuse(
            Status.UNSUPPORTED,
            "a parameter is named C1, C2, ... (or C_1, ...): those names are used for the constants of integration",
            "constant_name",
        )

    equations, functions, _ = _function_form(problem)
    try:
        with deadline(0.5 * limits.timeout_seconds):
            raw = sp.dsolve(
                equations if len(equations) > 1 else equations[0],
                functions if len(functions) > 1 else functions[0],
            )
    except SolverTimeout:
        mark("solve", clock)
        return refuse(
            Status.RESOURCE_LIMIT,
            "SymPy did not finish within the time limit. That does not show that no closed form "
            "exists; a numerical or series method can still be tried.",
            "solver_timeout",
        )
    except (
        NotImplementedError,
        ValueError,
        TypeError,
        KeyError,
        RecursionError,
        AttributeError,
        ArithmeticError,
    ):
        mark("solve", clock)
        return refuse(
            Status.UNSUPPORTED,
            "SymPy found no closed form for this equation. That does not show that none exists; "
            "a numerical or series method can still be tried.",
            "no_closed_form",
        )
    clock = mark("solve", clock)

    candidates = _candidates(raw, problem, functions)[:MAX_CANDIDATES]
    diagnostics.candidates = len(candidates)
    if not candidates:
        return refuse(
            Status.UNSUPPORTED,
            "SymPy returned a form Minerva cannot check, so nothing is shown.",
            "unreadable_result",
        )

    shown: list[OdeSolutionItem] = []
    dropped = 0
    notes: list[str] = []
    for cand in candidates:
        if sum(sp.count_ops(e) for e in cand.expressions) > MAX_RESULT_OPS:
            dropped += 1
            notes.append("A result was too large to check and is not shown.")
            continue
        constants = constants_of(*cand.expressions)
        verification, residual = _check(problem, cand, constants, symbolic=True)
        if not residual.passed:
            dropped += 1
            continue
        if cand.relation is not None and len(constants) == 1 == sum(problem.orders.values()):
            verification.independent_constants = (
                1 if _implicit_depends_on_constant(problem, cand.relation, constants[0]) else 0
            )
        clock = mark("verify", clock)
        variants: list[tuple[Candidate, list[sp.Symbol], OdeVerification, bool]] = []
        if problem.conditions:
            if cand.kind == "unevaluated":
                notes.append(
                    "The result contains an integral that SymPy could not evaluate; the conditions "
                    "were not applied to it."
                )
                continue
            fitted, why = _fix_constants(problem, cand, constants)
            if not fitted:
                notes.append(why[0].upper() + why[1:] + ".")
                continue
            for particular in fitted:
                left = constants_of(*particular.expressions)
                check, particular_residual = _check(problem, particular, left, symbolic=True)
                if not particular_residual.passed:
                    dropped += 1
                    continue
                variants.append((particular, left, check, True))
        else:
            variants.append((cand, constants, verification, False))
        clock = mark("conditions", clock)
        for variant, left, check, fixed in variants:
            conditions = _condition_checks(problem, variant, left)
            if not all(c.satisfied for c in conditions):
                dropped += 1
                continue
            check.passed = True
            rank = check.independent_constants
            shown.append(
                OdeSolutionItem(
                    kind=variant.kind,
                    equations_latex=_show(problem, variant),
                    rhs_text=[str(o) for o in variant.outputs] if variant.relation is None else [],
                    constants=[c.name for c in left],
                    family=_family(problem, left, rank, fixed),
                    evaluated=variant.kind != "unevaluated",
                    verification=check,
                    conditions=conditions,
                    restrictions_latex=_restrictions(variant),
                )
            )

    diagnostics.verified = len(shown)
    diagnostics.dropped = dropped
    if shown:
        diagnostics.solution_kind = shown[0].kind
        diagnostics.max_residual = max((s.verification.max_residual or 0.0) for s in shown)
    notes.append(
        "Singular solutions are not searched, and branch cuts of roots and logarithms are not analysed."
    )
    if dropped:
        notes.append(f"{dropped} candidate(s) from SymPy failed their checks and are not shown.")
    if not shown:
        status = Status.NOT_VERIFIED if dropped else Status.UNSUPPORTED
        message = (
            "SymPy proposed a solution but it did not pass Minerva's independent checks, so none is shown."
            if dropped
            else "No solution could be shown. " + " ".join(n for n in notes[:1])
        )
        diagnostics.failure_reason = "not_verified" if dropped else "no_solution"
        return finish(
            OdeSolution(
                status=status,
                message=message,
                method="closed_form",
                interpretation=meaning,
                notes=notes,
                dropped=dropped,
            )
        )
    weakest = OdeVerification(
        points_checked=min(s.verification.points_checked for s in shown),
        points_agreed=min(s.verification.points_agreed for s in shown),
        points_skipped=max(s.verification.points_skipped for s in shown),
        max_residual=max((s.verification.max_residual or 0.0) for s in shown),
        symbolic_zero=True if all(s.verification.symbolic_zero is True for s in shown) else None,
        independent_constants=min((s.verification.independent_constants or 0) for s in shown),
        passed=all(s.verification.passed for s in shown),
    )
    return finish(
        OdeSolution(
            status=Status.OK,
            message="",
            method="closed_form",
            interpretation=meaning,
            verification=weakest,
            solutions=shown,
            dropped=dropped,
            notes=notes,
        )
    )
