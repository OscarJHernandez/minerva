"""Noncommutative matrix calculus: parse, recognise the structure, pick a method, verify.

Entry point `matrix_calculus`. Matrices are declared (an abstract size or explicit entries) and
become noncommutative symbols, so the parsers keep the order of every product. The method is chosen
by `methods.select` from the structure alone; the result is returned only if the method's own
verification passed, and an equation or a representation is never presented as an evaluated integral.
"""

from __future__ import annotations

import time
from itertools import combinations

import sympy as sp

from ..build import Builder
from ..errors import LimitError, MinervaMathError, ParseError, UnsupportedInput
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
    MatrixVerification,
)
from .verify import Evaluator, Setup

DEADLINE_FRACTION = 0.8
RESULT_CHARS = 20_000


def matrix_calculus(data: MatrixCalculusInput, limits: Limits) -> MatrixCalculusResult:
    started = time.monotonic()
    deadline = started + DEADLINE_FRACTION * limits.timeout_seconds
    try:
        return _run(data, limits, deadline)
    except LimitError as exc:
        return _fail(data, Status.RESOURCE_LIMIT, str(exc))
    except (UnsupportedMatrixExpression, UnsupportedInput) as exc:
        return _fail(data, Status.UNSUPPORTED, str(exc))
    except MinervaMathError as exc:
        return _fail(data, Status.PARSE_ERROR, str(exc))
    except RecursionError:
        return _fail(data, Status.RESOURCE_LIMIT, "The expression is nested too deeply.")


def _fail(data: MatrixCalculusInput, status: Status, message: str) -> MatrixCalculusResult:
    return MatrixCalculusResult(status=status, action=data.action, message=message)


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


def _run(data: MatrixCalculusInput, limits: Limits, deadline: float) -> MatrixCalculusResult:
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

    expr = parse_expression(data.expression, limits, noncommutative=names)
    terms = to_terms(expr, nameset)
    shape = shape_of(terms, shapes)
    used = matrices_in(terms)

    comm, labels, cliques = _commutation(data, explicit, shapes)
    invertible = _invertible(data, explicit, shapes)

    var, lower, upper = _variable_and_bounds(data, limits, nameset, terms, explicit)
    if lower is not None and upper is not None and lower == upper:
        raise ParseError("the two bounds are equal")

    symbols = _scalar_symbols(terms, explicit, [x for x in (lower, upper) if x is not None], var)
    setup = Setup(shapes, explicit, comm, cliques)
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
    method, attempt, blocked, reason = select(problem)
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
        assumptions.append(f"all matrices are constant in {to_latex(p.var)}")
    message = {
        "explicit": "Evaluated and verified.",
        "expression": "Transformed and verified.",
        "equation": "Reduced to a verified equation; the integral itself was not evaluated.",
        "representation": (
            "Written exactly through a block-matrix exponential and verified; "
            "it is a representation, not an evaluated result."
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
            explicit[x] * explicit[y] - explicit[y] * explicit[x]
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
    return Commutation(frozenset(pairs)), labels, cliques


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
) -> tuple[sp.Symbol | None, sp.Expr | None, sp.Expr | None]:
    if data.action not in ("differentiate", "integrate"):
        if data.variable or data.lower or data.upper:
            raise ParseError("a variable and bounds are only used to differentiate or integrate")
        return None, None, None
    if not data.variable:
        raise ParseError("give the variable to differentiate or integrate with respect to")
    flags = {item: True for item in data.expression.assumptions.get(data.variable, [])}
    var = sp.Symbol(data.variable, **flags)
    if any(var in m.free_symbols for m in explicit.values()):
        raise ParseError(
            "matrix entries must not depend on the variable: the matrices are constant"
        )
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
