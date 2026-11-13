"""Noncommutative matrix calculus: parse, recognise the structure, pick a method, verify.

Entry point `matrix_calculus`. Matrices are declared (an abstract size or explicit entries) and
become noncommutative symbols, so the parsers keep the order of every product. The method is chosen
by `methods.select` from the structure alone; the result is returned only if the method's own
verification passed, and an equation or a representation is never presented as an evaluated integral.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from itertools import combinations
from typing import Literal

import sympy as sp

from ..build import Builder
from ..errors import LimitError, MinervaMathError, ParseError
from ..latexprint import to_latex
from ..models import ExpressionInput, Limits, Status
from .expressions import (
    Commutation,
    Exp,
    Terms,
    UnsupportedMatrixExpression,
    render_terms,
    shape_of,
    to_terms,
)
from .methods import Outcome, Problem, matrices_in, select
from .models import (
    MatrixBlocked,
    MatrixCalculusInput,
    MatrixCalculusResult,
    MatrixDiagnostics,
    MatrixMethodTried,
    MatrixStage,
    MatrixVerification,
)
from .verify import Evaluator, Setup, Telemetry, start_telemetry

DEADLINE_FRACTION = 0.8
RESULT_CHARS = 20_000


@dataclass
class _Trace:
    """What the diagnostics need to know about a computation while it is running."""

    started: float
    record: Telemetry
    problem: Problem | None = None
    labels: dict[str, str] = field(default_factory=dict)
    outcome: Outcome | None = None
    parse_s: float | None = None
    select_s: float | None = None


def matrix_calculus(data: MatrixCalculusInput, limits: Limits) -> MatrixCalculusResult:
    started = time.monotonic()
    trace = _Trace(started, start_telemetry())
    deadline = started + DEADLINE_FRACTION * limits.timeout_seconds
    try:
        result = _run(data, limits, deadline, trace)
    except LimitError as exc:
        result = _fail(data, Status.RESOURCE_LIMIT, str(exc))
    except UnsupportedMatrixExpression as exc:
        result = _fail(data, Status.UNSUPPORTED, str(exc))
    except MinervaMathError as exc:
        result = _fail(data, Status.PARSE_ERROR, str(exc))
    except RecursionError:
        result = _fail(data, Status.RESOURCE_LIMIT, "The expression is nested too deeply.")
    result.diagnostics = _diagnostics(data, result, trace)
    return result


def _fail(data: MatrixCalculusInput, status: Status, message: str) -> MatrixCalculusResult:
    return MatrixCalculusResult(status=status, action=data.action, message=message)


# -- diagnostics ------------------------------------------------------------------------------
VERIFICATION_IDS = {
    "exp_explicit": "expm_samples",
    "exp_merge_commuting": "expm_samples+series",
    "exp_derivative": "central_difference+series",
    "exp_frechet_derivative": "central_difference+quadrature+series",
    "integrate_polynomial": "quadrature+series",
    "exp_integral_invertible": "central_difference+series",
    "duhamel": "quadrature+series",
    "sylvester_reduction": "quadrature+residual",
    "block_exponential": "quadrature+residual",
    "sylvester_equation": "quadrature+series",
    "evolution_commuting": "central_difference+initial_condition+series",
    "evolution_numeric": "rk4+backward+liouville",
}


def _failure_code(result: MatrixCalculusResult) -> str | None:
    status = result.status
    if status is Status.OK:
        return None
    if status is Status.RESOURCE_LIMIT:
        text = result.message.lower()
        return "time_budget" if "time" in text else "size_limit"
    if status is Status.NOT_VERIFIED:
        return (
            "exact_disagreement"
            if result.verification and (result.verification.symbolic_zero is False)
            else "numeric_disagreement"
        )
    return status.value


def _source(
    statements: list[str], labels: dict[str, str]
) -> Literal["not_used", "declared", "from_entries", "mixed"]:
    return _source_kind({labels[a] for a in statements if a in labels})


def _source_kind(kinds: set[str]) -> Literal["not_used", "declared", "from_entries", "mixed"]:
    if kinds == {"declared"}:
        return "declared"
    if kinds == {"from the entries"}:
        return "from_entries"
    return "mixed" if kinds else "not_used"


def _diagnostics(
    data: MatrixCalculusInput, result: MatrixCalculusResult, trace: _Trace
) -> MatrixDiagnostics:
    now = time.monotonic()
    record = trace.record
    verify_s = record.numeric_s + record.exact_s
    stages: list[MatrixStage] = []
    if trace.parse_s is not None:
        stages.append(MatrixStage(name="parse", ms=round(trace.parse_s * 1000, 2)))
    if trace.select_s is not None:
        transform = max(0.0, trace.select_s - verify_s)
        stages.append(MatrixStage(name="transform", ms=round(transform * 1000, 2)))
        stages.append(MatrixStage(name="verify_numeric", ms=round(record.numeric_s * 1000, 2)))
        stages.append(MatrixStage(name="verify_exact", ms=round(record.exact_s * 1000, 2)))
    stages.append(MatrixStage(name="total", ms=round((now - trace.started) * 1000, 2)))
    p = trace.problem
    used_shapes = sorted({p.setup.shapes[n] for n in matrices_in(p.terms)}) if p is not None else []
    assumptions = list(result.assumptions_used)
    labelled = [a.rsplit(" (", 1)[0] for a in assumptions]
    invertible = {
        "from the entries" if a.endswith("(from the entries)") else "declared"
        for a in assumptions
        if "is invertible (" in a
    }
    v = result.verification
    return MatrixDiagnostics(
        trace_id=data.trace_id,
        method=result.method,
        methods_tried=[
            MatrixMethodTried(id=m, outcome=o)
            for m, o in (p.scratch.get("tried", []) if p is not None else [])
        ],
        dimensions=used_shapes,
        matrices=len(data.matrices),
        explicit=sum(1 for d in data.matrices if d.entries is not None),
        time_dependent=len(p.setup.dependent) if p is not None else 0,
        commutation=_source(labelled, trace.labels),
        invertibility=_source_kind(invertible),
        stages=stages,
        digits=record.digits,
        tolerance=record.tolerance,
        max_residual=record.max_residual,
        points_checked=v.points_checked if v else 0,
        points_agreed=v.points_agreed if v else 0,
        verification=VERIFICATION_IDS.get(result.method or "", "") if v else "",
        fallbacks=list(record.fallbacks),
        failure_reason=_failure_code(result),
    )


def _parse_scalar(
    text: str, data: MatrixCalculusInput, limits: Limits, names: frozenset[str]
) -> sp.Expr:
    from ..ops import parse_expression  # local import: ops imports this package lazily

    expr = parse_expression(
        ExpressionInput(
            input=text, format=data.expression.format, assumptions=data.expression.assumptions
        ),
        limits,
    )
    clash = sorted(s.name for s in expr.free_symbols if s.name in names)
    if clash:
        raise ParseError(f"{clash[0]!r} is a matrix and cannot appear in a scalar entry or bound")
    return expr


def _run(
    data: MatrixCalculusInput, limits: Limits, deadline: float, trace: _Trace
) -> MatrixCalculusResult:
    from ..ops import parse_expression  # local import: ops imports this package lazily

    names = [d.name for d in data.matrices]
    if len(set(names)) != len(names):
        raise ParseError("each matrix must be declared once")
    probe = Builder(limits=limits)
    for name in names:
        probe.symbol(name)  # validates the name (shape, reserved words)
    nameset = frozenset(names)
    if data.variable in nameset:
        raise ParseError("the variable cannot be a declared matrix")

    # explicit matrices: entries are scalar expressions (possibly with parameters)
    explicit: dict[str, sp.Matrix] = {}
    for decl in data.matrices:
        if decl.entries is not None:
            explicit[decl.name] = sp.Matrix(
                [[_parse_scalar(e, data, limits, nameset) for e in row] for row in decl.entries]
            )
    shapes = {d.name: d.shape for d in data.matrices}

    dependent = _dependent(data, explicit)
    source = _time_arguments(data, dependent)
    expr = parse_expression(source, limits, noncommutative=names)
    terms = to_terms(expr, nameset)
    shape = shape_of(terms, shapes)
    used = matrices_in(terms)

    var, lower, upper = _variable_and_bounds(data, limits, nameset, terms, explicit, dependent)
    if lower is not None and upper is not None and lower == upper:
        raise ParseError("the two bounds are equal")
    comm, labels, cliques = _commutation(data, explicit, shapes, dependent, var)
    invertible = _invertible(data, explicit, shapes)

    symbols = _scalar_symbols(terms, explicit, [x for x in (lower, upper) if x is not None], var)
    setup = Setup(shapes, explicit, comm, cliques, var, frozenset(dependent & used))
    trace.labels = labels
    trace.parse_s = time.monotonic() - trace.started
    problem = Problem(
        data.action,
        terms,
        shape,
        setup,
        Evaluator(symbols),
        invertible,
        var,
        lower,
        upper,
        deadline,
    )
    trace.problem = problem
    selecting = time.monotonic()
    method, attempt, blocked, reason = select(problem)
    trace.select_s = time.monotonic() - selecting
    trace.outcome = attempt.outcome
    base = MatrixCalculusResult(
        status=Status.OK,
        action=data.action,
        input_latex=render_terms(terms),
        features=_features(problem, used),
        blocked=blocked,
        structure=_structure(problem),
        notes=_scalar_notes(symbols),
    )
    if method is None or attempt.outcome is None:
        return _no_outcome(data, base, blocked, reason)
    return _with_outcome(problem, base, method.id, method.title, attempt.outcome, labels)


def _no_outcome(
    data: MatrixCalculusInput,
    base: MatrixCalculusResult,
    blocked: list[MatrixBlocked],
    reason: str | None,
) -> MatrixCalculusResult:
    if reason:
        return base.model_copy(update={"status": Status.UNSUPPORTED, "message": _sentence(reason)})
    if blocked:
        needs = "; ".join(f"{b.title} needs {b.needs}" for b in blocked)
        return base.model_copy(
            update={
                "status": Status.CONDITIONS_NOT_SATISFIED,
                "message": (
                    "Nothing was changed because a step needs a condition that is not "
                    f"established: {needs}. Declare it if it holds."
                ),
            }
        )
    if data.action == "simplify":
        return base.model_copy(
            update={
                "status": Status.UNCHANGED,
                "message": (
                    "No verified simplification was found within the supported identities; this "
                    "does not mean that none exists."
                ),
            }
        )
    return base.model_copy(
        update={
            "status": Status.UNSUPPORTED,
            "message": "No supported method applies to this structure.",
        }
    )


def _with_outcome(
    p: Problem,
    base: MatrixCalculusResult,
    method_id: str,
    method_title: str,
    outcome: Outcome,
    labels: dict[str, str],
) -> MatrixCalculusResult:
    verification = MatrixVerification(
        method=outcome.numeric_method,
        points_checked=outcome.count.checked,
        points_agreed=outcome.count.agreed,
        passed=outcome.count.passed,
        symbolic_zero=outcome.symbolic,
        symbolic_method=outcome.symbolic_method,
        note=outcome.note,
    )
    common = base.model_copy(update={"method": method_id, "verification": verification})
    if not outcome.count.passed or outcome.symbolic is False:
        why = (
            "an exact check disagreed"
            if outcome.symbolic is False
            else "the numerical check did not agree at enough sample points"
        )
        return common.model_copy(
            update={
                "status": Status.NOT_VERIFIED,
                "message": (
                    f"A candidate was produced but could not be verified ({why}), so it is not shown."
                ),
            }
        )
    if outcome.result_latex is not None and len(outcome.result_latex) > RESULT_CHARS:
        raise LimitError("The result is too large to show.")
    assumptions = [f"{a} ({labels[a]})" if a in labels else a for a in outcome.assumptions]
    if any(s != p.var for s in p.ev.symbols):
        assumptions.append("scalar symbols are complex unless declared otherwise")
    if p.var is not None:
        moving = sorted(p.setup.dependent)
        fixed = sorted(matrices_in(p.terms) - p.setup.dependent)
        if moving:
            assumptions.append(
                f"{', '.join(moving)} {'depends' if len(moving) == 1 else 'depend'} on "
                f"{to_latex(p.var)}"
            )
        if fixed:
            assumptions.append(
                f"{', '.join(fixed)} {'is' if len(fixed) == 1 else 'are'} constant in "
                f"{to_latex(p.var)}"
            )
    message = {
        "explicit": "Evaluated and verified.",
        "expression": "Transformed and verified.",
        "equation": "Reduced to a verified equation; the integral itself was not evaluated.",
        "representation": (
            "Written exactly (for example through a block-matrix exponential or an integral) and verified; "
            "it is a representation, not an evaluated result."
        ),
        "numeric": (
            "Computed numerically on the stated span and verified by independent checks; this is "
            "a numerical approximation, not a closed form."
        ),
    }[outcome.kind]
    return common.model_copy(
        update={
            "status": Status.OK,
            "method_title": method_title,
            "result_kind": outcome.kind,
            "result_latex": outcome.result_latex,
            "result_entries": outcome.entries,
            "equation_latex": outcome.equation_latex,
            "evaluated": outcome.evaluated,
            "assumptions_used": assumptions,
            "steps": outcome.steps,
            "message": message,
            "notes": [*outcome.notes, *base.notes],
        }
    )


def _sentence(text: str) -> str:
    text = text[0].upper() + text[1:]
    return text if text.endswith(".") else text + "."


# -- declarations -----------------------------------------------------------------------------
def _commutation(
    data: MatrixCalculusInput,
    explicit: dict[str, sp.Matrix],
    shapes: dict[str, tuple[int, int]],
    dependent: set[str],
    var: sp.Symbol | None,
) -> tuple[Commutation, dict[str, str], list[list[str]]]:
    declared: set[frozenset[str]] = set()
    for a, b in data.commuting:
        if a not in shapes or b not in shapes:
            raise ParseError("a commuting pair names a matrix that is not declared")
        if a == b:
            raise ParseError("a matrix always commutes with itself; name two different matrices")
        declared.add(frozenset((a, b)))
    pairs: set[frozenset[str]] = set(declared)
    labels: dict[str, str] = {}
    for pair in declared:
        x, y = sorted(pair)
        labels[f"{x}{y} = {y}{x}"] = "declared"
    for x, y in combinations(sorted(explicit), 2):
        diff = (
            _commutator_at_all_times(explicit, x, y, dependent, var)
            if shapes[x] == shapes[y]
            else None
        )
        if diff is None:
            continue
        zero = all(sp.simplify(e) == 0 for e in diff)
        key = frozenset((x, y))
        if zero and key not in declared:
            pairs.add(key)
            labels[f"{x}{y} = {y}{x}"] = "from the entries"
        elif not zero and key in declared:
            raise ParseError(f"{x} and {y} are declared to commute but their entries do not")
    # commutation need not be transitive: connected groups must be fully connected
    groups = _components(set(shapes), pairs)
    cliques: list[list[str]] = []
    for group in groups:
        if len(group) < 2:
            continue
        if any(frozenset(pq) not in pairs for pq in combinations(group, 2)):
            raise UnsupportedMatrixExpression(
                "declare the commutation of every pair in a group of matrices "
                f"({', '.join(sorted(group))}); a chain such as AB=BA, BC=CB does not imply AC=CA"
            )
        if len({shapes[n] for n in group}) != 1 or shapes[group[0]][0] != shapes[group[0]][1]:
            raise ParseError("matrices declared to commute must be square and of the same size")
        cliques.append(sorted(group))
    selfs = _self_commuting(data, explicit, dependent, var, labels)
    return Commutation(frozenset(pairs), selfs), labels, cliques


def _commutator_at_all_times(
    explicit: dict[str, sp.Matrix],
    x: str,
    y: str,
    dependent: set[str],
    var: sp.Symbol | None,
) -> sp.Matrix:
    """[X, Y] for explicit matrices. A time-dependent matrix is compared at two different times,
    because the derivative of a commutator that vanishes only at equal times need not vanish."""
    other = sp.Dummy("tau")
    right = explicit[y]
    if var is not None and x in dependent and y in dependent:
        right = right.subs(var, other)
    return explicit[x] * right - right * explicit[x]


def _self_commuting(
    data: MatrixCalculusInput,
    explicit: dict[str, sp.Matrix],
    dependent: set[str],
    var: sp.Symbol | None,
    labels: dict[str, str],
) -> frozenset[str]:
    out: set[str] = set()
    for name in data.self_commuting:
        if name not in dependent:
            raise ParseError(
                f"{name} is not a time-dependent matrix, so it has no self-commutation"
            )
        out.add(name)
        labels[f"{name}(t1) {name}(t2) = {name}(t2) {name}(t1) at all times"] = "declared"
    for name in sorted(dependent & set(explicit)):
        if var is None:
            continue
        other = sp.Dummy("tau")
        shifted = explicit[name].subs(var, other)
        diff = explicit[name] * shifted - shifted * explicit[name]
        zero = all(sp.simplify(e) == 0 for e in diff)
        key = f"{name}(t1) {name}(t2) = {name}(t2) {name}(t1) at all times"
        if zero and name not in out:
            out.add(name)
            labels[key] = "from the entries"
        elif not zero and name in out:
            raise ParseError(
                f"{name} is declared to commute with itself at all times but its entries do not"
            )
    return frozenset(out)


def _dependent(data: MatrixCalculusInput, explicit: dict[str, sp.Matrix]) -> set[str]:
    """Matrices that are functions of the variable: declared, or visible in explicit entries."""
    found: set[str] = set()
    for decl in data.matrices:
        if decl.depends_on is None:
            continue
        if data.variable is None or decl.depends_on != data.variable:
            raise ParseError(
                f"{decl.name} is declared to depend on {decl.depends_on}, which is not the "
                "variable of this operation"
            )
        found.add(decl.name)
    if data.variable is not None:
        for name, m in explicit.items():
            if any(sym.name == data.variable for sym in m.free_symbols):
                found.add(name)
    if found and data.action not in ("differentiate", "evolve"):
        raise UnsupportedMatrixExpression(
            "time-dependent matrices are supported for differentiate and evolve only"
        )
    return found


def _time_arguments(data: MatrixCalculusInput, dependent: set[str]) -> ExpressionInput:
    """Read A(t) as the matrix A when A is declared time dependent; refuse to read it as t times A
    otherwise (implicit multiplication would silently change the meaning)."""
    text = data.expression.input
    if data.variable is None:
        return data.expression
    var = re.escape(data.variable)
    for decl in data.matrices:
        pattern = re.compile(
            r"(?<![A-Za-z0-9\\])"
            + re.escape(decl.name)
            + r"\s*(?:\\left\s*\(|\()\s*"
            + var
            + r"\s*(?:\\right\s*\)|\))"
        )
        if not pattern.search(text):
            continue
        if decl.name not in dependent:
            raise ParseError(
                f"{decl.name}({data.variable}) would be read as {data.variable} times "
                f"{decl.name}; write {data.variable}{decl.name} for the product, or declare "
                f"{decl.name} as depending on {data.variable}"
            )
        text = pattern.sub(decl.name, text)
    return data.expression.model_copy(update={"input": text})


def _components(nodes: set[str], pairs: set[frozenset[str]]) -> list[list[str]]:
    parent = {n: n for n in nodes}

    def find(n: str) -> str:
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    for pair in pairs:
        a, b = sorted(pair)
        parent[find(a)] = find(b)
    groups: dict[str, list[str]] = {}
    for n in sorted(nodes):
        groups.setdefault(find(n), []).append(n)
    return list(groups.values())


def _invertible(
    data: MatrixCalculusInput,
    explicit: dict[str, sp.Matrix],
    shapes: dict[str, tuple[int, int]],
) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in data.invertible:
        if name not in shapes:
            raise ParseError("an invertible matrix is not declared")
        if shapes[name][0] != shapes[name][1]:
            raise ParseError(f"{name} is not square, so it cannot be invertible")
        if name in explicit and sp.simplify(explicit[name].det()) == 0:
            raise ParseError(f"{name} is declared invertible but its determinant is zero")
        out[name] = "declared"
    for name, m in explicit.items():
        if name not in out and m.rows == m.cols and sp.simplify(m.det()).is_zero is False:
            out[name] = "from the entries"
    return out


def _variable_and_bounds(
    data: MatrixCalculusInput,
    limits: Limits,
    names: frozenset[str],
    terms: Terms,
    explicit: dict[str, sp.Matrix],
    dependent: set[str],
) -> tuple[sp.Symbol | None, sp.Expr | None, sp.Expr | None]:
    if data.action not in ("differentiate", "integrate", "evolve"):
        if data.variable or data.lower or data.upper:
            raise ParseError(
                "a variable and bounds are only used to differentiate, integrate or evolve"
            )
        return None, None, None
    if not data.variable:
        raise ParseError("give the variable (time) of this operation")
    flags = {item: True for item in data.expression.assumptions.get(data.variable, [])}
    var = sp.Symbol(data.variable, **flags)
    if data.action == "integrate" and dependent:
        raise UnsupportedMatrixExpression(
            "integrals of time-dependent matrices are not supported; matrix entries must not "
            "depend on the variable of integration"
        )
    if data.action == "evolve":
        lower = _parse_scalar(data.lower, data, limits, names) if data.lower else sp.Integer(0)
        upper = _parse_scalar(data.upper, data, limits, names) if data.upper else None
        for bound in (lower, upper):
            if bound is not None and (
                bound.has(sp.oo, -sp.oo, sp.zoo) or var in bound.free_symbols
            ):
                raise ParseError("the time bounds must be finite and must not contain the variable")
        return var, lower, upper
    if data.action == "differentiate":
        if data.lower or data.upper:
            raise ParseError("bounds are only used to integrate")
        return var, None, None
    if (data.lower is None) != (data.upper is None):
        raise ParseError("give both bounds for a definite integral, or neither")
    if data.lower is None or data.upper is None:
        return var, None, None
    lower = _parse_scalar(data.lower, data, limits, names)
    upper = _parse_scalar(data.upper, data, limits, names)
    for bound in (lower, upper):
        if bound.has(sp.oo, -sp.oo, sp.zoo):
            raise UnsupportedMatrixExpression("infinite bounds are not supported")
        if var in bound.free_symbols:
            raise ParseError("a bound cannot contain the variable of integration")
    return var, lower, upper


def _scalar_symbols(
    terms: Terms,
    explicit: dict[str, sp.Matrix],
    bounds: list[sp.Expr],
    var: sp.Symbol | None,
) -> list[sp.Symbol]:
    found: set[sp.Symbol] = set()
    for t in terms:
        found |= t.coeff.free_symbols
        for f in t.factors:
            if isinstance(f, Exp):
                for _, c in f.arg:
                    found |= c.free_symbols
    for m in explicit.values():
        found |= m.free_symbols
    for b in bounds:
        found |= b.free_symbols
    if var is not None:
        found.add(var)
    return sorted((s for s in found if isinstance(s, sp.Symbol)), key=lambda s: s.name)


def _scalar_notes(symbols: list[sp.Symbol]) -> list[str]:
    return [
        f"{s.name} is not a declared matrix, so it is treated as a scalar"
        for s in symbols
        if s.name[0].isupper()
    ]


# -- description of what was recognised -------------------------------------------------------
def _features(p: Problem, used: set[str]) -> list[str]:
    feats = [f"matrix:{p.action}"]
    exps = [f for t in p.terms for f in t.factors if isinstance(f, Exp)]
    if exps:
        feats.append("matrix:exp")
    exp_names = sorted({n for f in exps for n in f.names})
    if any(not p.comm.commutes(a, b) for a, b in combinations(exp_names, 2)):
        feats.append("matrix:noncommuting")
    if p.explicit and used <= set(p.explicit):
        feats.append("matrix:explicit")
    if "reduction" in p.scratch and not isinstance(p.scratch["reduction"], str):
        feats.append("matrix:sylvester-form")
    selected = {m for m, o in p.scratch.get("tried", []) if o == "selected"}
    if "exp_frechet_derivative" in selected:
        feats.append("matrix:frechet")
    if "duhamel" in selected:
        feats.append("matrix:duhamel")
    if p.setup.dependent:
        feats.append("matrix:parameter-dependent")
    if p.action == "evolve":
        feats.append("matrix:evolution")
    return feats


def _structure(p: Problem) -> str:
    exps = [f for t in p.terms for f in t.factors if isinstance(f, Exp)]
    names = sorted(matrices_in(p.terms))
    parts = [
        f"Sum of {len(p.terms)} terms" if len(p.terms) > 1 else "A single term",
        f"in the matrices {', '.join(names)}",
    ]
    if exps:
        parts.append(f"with {len(exps)} matrix exponential{'s' if len(exps) != 1 else ''}")
    if len({n for f in exps for n in f.names}) > 1 or len(names) > 1:
        pairs = [(a, b) for a, b in combinations(names, 2)]
        shapes = p.setup.shapes
        free = [
            f"{a}{b}"
            for a, b in pairs
            if not p.comm.commutes(a, b) and shapes[a] == shapes[b] and shapes[a][0] == shapes[a][1]
        ]
        if free:
            parts.append("commutation is not assumed for " + ", ".join(free))
        elif all(p.comm.commutes(a, b) for a, b in pairs):
            parts.append("all of these matrices commute")
    text = " ".join(parts[:2]) + (", " + ", ".join(parts[2:]) if parts[2:] else "") + "."
    extra = p.scratch.get("structure")
    return f"{text} {extra}" if extra else text
