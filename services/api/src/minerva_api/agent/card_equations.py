"""The equations a card lists, so that each one can be numbered and referred to.

A card with one primary result (`result_latex`) is numbered as a whole (`equations.number_blocks`). A card
that is a *list* of equations (the Christoffel symbols of a metric, the equations of motion of a Lagrangian,
the constants found by a search) has no single formula; before this module its components had no numbers at
all and could not be named in a later message. Each reader below returns the card's listed equations in the
order the card shows them, as `(key, latex)`: the key is stable for the card (the client matches a row to its
number by it) and the latex is exactly what the row shows.

A number is a name, never a verification. `@eN` on a component substitutes its LaTeX exactly, but nothing
here links the component to the verification or the restrictions of the card it came from, so a result built
on one says that (`refs.py`, `UNLINKED_INPUT`).

Every card type must be classified in exactly one of the three sets below
(`test_every_card_type_is_classified_for_equation_numbers`): a new card type has to say which it is.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

Listed = list[tuple[str, str]]

# Cards that keep their whole tool result under `data["result"]` (a step, a matrix calculation) instead of at the
# top of `data`. Before `result_formula` the numbering and the reference table read only the top level, so a
# successful `substitute` or `rewrite_part` never got a number although `step` is a single-result card (#133).
# `sums`, `expectation` and `distribution` are the same case (`data["result"]`). `approximation` keeps its result
# under `data["report"]` (`_REPORT_RESULT_CARDS`), a different shape with its own gate (`_approximation_formula`).
NESTED_RESULT_CARDS = frozenset({"step", "matrix", "sums", "expectation", "distribution"})
_REPORT_RESULT_CARDS = frozenset({"approximation"})
# `result_kind` values of a matrix card that are not an evaluated expression: an equation, a representation, or a
# numerical approximation. Such a result is never numbered or given an id as if it were a closed form.
_NOT_EVALUATED_KINDS = frozenset({"equation", "representation", "numeric"})


def _nested(card_type: object, data: dict[str, Any]) -> dict[str, Any] | None:
    inner = data.get("result")
    return inner if card_type in NESTED_RESULT_CARDS and isinstance(inner, dict) else None


def result_formula(card_type: object, data: dict[str, Any]) -> str | None:
    """The one result formula of a card that can be numbered as a whole, or None.

    One reader for every place that needs it (`equations.number_blocks`, `number_stored`, `build_table`,
    `refs.ResultStore.register` and the rebuild in `turn_context`), so a live table and one rebuilt from stored
    rows agree. The top-level `result_latex` of a card is returned as it is (the tool's own gate decided
    whether to write one). For a `step` or `matrix` card the formula lives in `data["result"]`, and counts only
    if that result is `ok`; a matrix result must also be `evaluated` (an equation, a representation, a block
    or integral form and a numerical approximation are not an evaluated result, so they have no number).
    """
    top = _text(data.get("result_latex"))
    if top is not None:
        return top
    if card_type in _REPORT_RESULT_CARDS:
        return _approximation_formula(data.get("report"))
    inner = _nested(card_type, data)
    if inner is None or inner.get("status") != "ok":
        return None
    if card_type == "matrix" and (
        inner.get("evaluated") is not True or inner.get("result_kind") in _NOT_EVALUATED_KINDS
    ):
        return None
    # An expectation that holds only given an independence the user declared. The number would be a name for
    # the formula alone: a later `@eN` could not carry the premise (this card has no result id), so it gets none.
    if card_type == "expectation" and (
        inner.get("evidence") == "conditional" or inner.get("given")
    ):
        return None
    return _text(inner.get("result_latex"))


def _approximation_formula(report: object) -> str | None:
    """The approximation of an accepted `approximation` card, or None.

    Accepted means `status == "ok"` (the engine shows a candidate only after a check that does not use the route
    that built it), evidence of a kind (`none` is never a result), and no check that failed. A candidate that was
    not verified, an unsupported request, a question to the user and an unchanged input list nothing. The number
    names the formula; the card says what was measured (an error is a measurement on sampled points, never a
    bound) and `@eN` of it carries no verification (a result with no result id, `refs.NO_ID_SUFFIX`).
    """
    if not isinstance(report, dict) or report.get("status") != "ok":
        return None
    if report.get("evidence") in (None, "none"):
        return None
    checks = report.get("checks")
    if isinstance(checks, list) and any(
        isinstance(c, dict) and c.get("passed") is False for c in checks
    ):
        return None
    return _text(report.get("result_latex"))


def result_restrictions(card_type: object, data: dict[str, Any]) -> list[str]:
    """Where the card's result is valid: its own `restrictions_latex` (top level, or inside a step's result)."""
    inner = _nested(card_type, data)
    for source in (data, inner or {}):
        found = source.get("restrictions_latex")
        if isinstance(found, list) and found:
            return [r for r in found if isinstance(r, str)]
    return []


# What a premise label ends with in `unverified_inputs` (`refs.py` words the note and the quote reason).
PREMISE_SUFFIX = ", a relation you gave as a premise"


def premise_labels(card_type: object, data: dict[str, Any]) -> list[str]:
    """The premises a conditional step result rests on, as the labels `unverified_inputs` carries."""
    return [f"${p}${PREMISE_SUFFIX}" for p in result_premises(card_type, data)]


def result_premises(card_type: object, data: dict[str, Any]) -> list[str]:
    """The relations the user gave that a step result holds *given* (`evidence == "conditional"`), as LaTeX."""
    inner = _nested(card_type, data)
    if inner is None or inner.get("evidence") != "conditional":
        return []
    given = inner.get("given")
    if not isinstance(given, list):
        return []
    return [str(g["latex"]) for g in given if isinstance(g, dict) and _text(g.get("latex"))]


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _rows(items: object, field: str, prefix: str) -> Listed:
    out: Listed = []
    if isinstance(items, list):
        for i, item in enumerate(items):
            latex = _text(item.get(field)) if isinstance(item, dict) else None
            if latex:
                out.append((f"{prefix}:{i}", latex))
    return out


def geometry(data: dict[str, Any]) -> Listed:
    """Each shown quantity: its scalar, or every component as the row shows it (`label = value`)."""
    out: Listed = []
    quantities = (data.get("result") or {}).get("quantities")
    for q in quantities if isinstance(quantities, list) else []:
        if not isinstance(q, dict) or q.get("status") != "ok":
            continue
        kind = str(q.get("kind", ""))
        scalar = _text(q.get("scalar_latex"))
        if scalar:
            out.append((kind, scalar))
            continue
        for c in q.get("components") or []:
            if not isinstance(c, dict):
                continue
            latex = _text(c.get("latex"))
            if not latex:
                continue
            indices = ",".join(str(i) for i in c.get("indices") or [])
            label = _text(c.get("label_latex"))
            shown = latex if kind == "geodesics" or not label else f"{label} = {latex}"
            out.append((f"{kind}:{indices}", shown))
    return out


def lagrangian(data: dict[str, Any]) -> Listed:
    report = data.get("report") or {}
    return [
        *_rows(report.get("equations"), "latex", "eq"),
        *_rows(report.get("momenta"), "latex", "mom"),
        *_rows(report.get("accelerations"), "latex", "acc"),
        *_rows(report.get("conserved"), "expression_latex", "cons"),
    ]


def invariants(data: dict[str, Any]) -> Listed:
    return _rows((data.get("report") or {}).get("found"), "expression_latex", "found")


def discovery(data: dict[str, Any]) -> Listed:
    report = data.get("report") or {}
    relations = (report.get("independence") or {}).get("relations")
    return [
        *_rows(report.get("constants"), "expression_latex", "const"),
        *_rows(relations, "latex", "rel"),
    ]


def solve(data: dict[str, Any]) -> Listed:
    """The factors and the exact roots of an accepted `solve` card, as the card shows them.

    `factor:i` is a factor, `root:i:j` an exact root of factor i, `rootrow:k` the exact form in row k of the table
    of roots (the same formula as a `root:` row is one number). Not numbered: the equation as read (an echo of the
    user's input), the numerical decimals and the verification lines. A card that was not accepted (`not_verified`
    still shows its factors) lists nothing.
    """
    report = data.get("report") or {}
    if report.get("status") != "ok":
        return []
    out: Listed = []
    factors = report.get("factors")
    for i, factor in enumerate(factors if isinstance(factors, list) else []):
        if not isinstance(factor, dict):
            continue
        latex = _text(factor.get("factor_latex"))
        if latex:
            out.append((f"factor:{i}", latex))
        roots = factor.get("exact_roots_latex")
        for j, root in enumerate(roots if isinstance(roots, list) else []):
            if (text := _text(root)) is not None:
                out.append((f"root:{i}:{j}", text))
    return [*out, *_rows(report.get("roots"), "exact_latex", "rootrow")]


# A solution of kind `explicit` or `implicit` that is `evaluated` is a closed form; an unevaluated integral, a
# series and a perturbation expansion are representations or truncations and are never presented as a result.
_CLOSED_FORM_KINDS = frozenset({"explicit", "implicit"})


def ode_solution(data: dict[str, Any]) -> Listed:
    """`solution:s:j`: equation j of solution s, for a verified closed form only.

    Not numbered: restrictions, the checked conditions and the right-hand side text (the card's own bookkeeping).
    """
    report = data.get("report") or {}
    if report.get("status") != "ok":
        return []
    out: Listed = []
    solutions = report.get("solutions")
    for s, item in enumerate(solutions if isinstance(solutions, list) else []):
        if (
            not isinstance(item, dict)
            or item.get("kind") not in _CLOSED_FORM_KINDS
            or item.get("evaluated") is not True
        ):
            continue
        out += _rows(
            [{"latex": e} for e in item.get("equations_latex") or []], "latex", f"solution:{s}"
        )
    return out


def canonical_equilibria(items: object) -> list[dict[str, Any]]:
    """The equilibria of a qualitative analysis in one canonical order, whatever order the solver found them in.

    SymPy returns the solutions of `f = 0` in a dictionary order that depends on the version and the input, so a
    position in that list is not a name. Sorted by the text of the point (`y = 0`, `y' = 0`), then by the Jacobian
    and the characteristic polynomial; the sort is stable, so equal points keep their order. Plain code-point
    comparison, which the panel repeats exactly (`ode-equilibria.ts`).
    """
    found = [e for e in items if isinstance(e, dict)] if isinstance(items, list) else []
    return sorted(
        found,
        key=lambda e: (
            [str(p) for p in e.get("point_latex") or []],
            str(e.get("jacobian_latex") or ""),
            str(e.get("characteristic_latex") or ""),
        ),
    )


def canonical_eigenvalues(items: object) -> list[str]:
    """The exact eigenvalues of one equilibrium, sorted by their text (the solver's order is not canonical)."""
    return sorted(str(x) for x in items if _text(x)) if isinstance(items, list) else []


def _equilibrium_rows(items: object) -> Listed:
    out: Listed = []
    for i, e in enumerate(canonical_equilibria(items)):
        for j, point in enumerate(e.get("point_latex") or []):
            if (text := _text(point)) is not None:
                out.append((f"equil:{i}:point:{j}", text))
        if (text := _text(e.get("jacobian_latex"))) is not None:
            out.append((f"equil:{i}:jacobian", text))
        for j, eig in enumerate(canonical_eigenvalues(e.get("eigenvalues_latex"))):
            out.append((f"equil:{i}:eig:{j}", eig))
        if (text := _text(e.get("characteristic_latex"))) is not None:
            out.append((f"equil:{i}:char", text))
    return out


def ode_analysis(data: dict[str, Any]) -> Listed:
    """`solved:i`, `field:i` and, per equilibrium, `equil:i:point:j`, `equil:i:jacobian`, `equil:i:eig:j`, `equil:i:char`.

    `solved:i` is an equation solved for its highest derivative, `field:i` a direction-field formula. The
    equilibria are sorted first (`canonical_equilibria`; i is the position in that order, the eigenvalues of
    one equilibrium likewise), because the solver's order is not canonical. An equilibrium was confirmed by
    substituting it back, so its point rows are numbered; its Jacobian, eigenvalues and characteristic
    polynomial only when the qualitative part says the Jacobian was checked against numerical differences of the
    field (`jacobian_checked`). Not numbered: the numerical eigenvalues (decimals), the classification text, the
    interpretation (the user's own input, quoted), restrictions and notes. A card that was not accepted lists nothing.
    """
    report = data.get("report") or {}
    if report.get("status") != "ok":
        return []
    qualitative = report.get("qualitative")
    field = None
    equilibria: Listed = []
    if isinstance(qualitative, dict) and qualitative.get("status") == "ok":
        field = qualitative.get("field_latex")
        equilibria = _equilibrium_rows(qualitative.get("equilibria"))
        if qualitative.get("jacobian_checked") is not True:
            equilibria = [(k, v) for k, v in equilibria if ":point:" in k]
    return [
        *_rows([{"latex": e} for e in report.get("solved_latex") or []], "latex", "solved"),
        *_rows([{"latex": e} for e in field or []], "latex", "field"),
        *equilibria,
    ]


# ----------------------------------------------------------------------------------------------------------
# variation, statistics, expansion (#133, part 2 and 3, PR 2 of 3)

_GREEK = frozenset(
    {
        *("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa"),
        *("lambda", "mu", "nu", "xi", "rho", "sigma", "tau", "phi", "chi", "psi", "omega"),
    }
)


def name_latex(name: str) -> str:
    """A parameter name as the statistics panel writes it (`mu` -> `\\mu`, `lam` -> `\\lambda`): the same table."""
    if name == "lam":
        return "\\lambda"
    return f"\\{name}" if name in _GREEK else name


def variation(data: dict[str, Any]) -> Listed:
    """The first-variation rows of an accepted `variation` card, and its second variation.

    `el:<function>` (an Euler-Lagrange equation `EL = 0`; none for a null Lagrangian, whose equations are the
    identity `0 = 0`), `first` (the complete first variation), `stationarity:i`,
    `natural:<end>:<function>:<order>` (the card's own row key) and `second` (the quadratic form).
    A first-variation row needs the report to be `ok` and its finite-difference verification to have passed;
    the second variation needs its own deciding check to have passed. Not numbered: the functional as read (the
    user's input), the integrand of the variation, every integration by parts, the boundary form and endpoint
    terms (bookkeeping the card shows to be checked), and every numerical table.
    """
    report = data.get("report") or {}
    if report.get("status") != "ok":
        return []
    out: Listed = []
    first = report.get("first")
    if isinstance(first, dict) and (first.get("verification") or {}).get("passed") is True:
        if first.get("null_lagrangian") is not True:
            for eq in first.get("equations") or []:
                function = _text(eq.get("function")) if isinstance(eq, dict) else None
                latex = _text(eq.get("equation_latex")) if isinstance(eq, dict) else None
                if function and latex:
                    out.append((f"el:{function}", latex))
        if (latex := _text(first.get("first_variation_latex"))) is not None:
            out.append(("first", latex))
        if first.get("null_lagrangian") is not True:
            out += _rows(
                [{"latex": s} for s in first.get("stationarity_latex") or []],
                "latex",
                "stationarity",
            )
        for b in first.get("boundary") or []:
            latex = _text(b.get("natural_condition_latex")) if isinstance(b, dict) else None
            if latex:
                out.append((f"natural:{b.get('end')}:{b.get('function')}:{b.get('order')}", latex))
    second = report.get("second")
    if (
        isinstance(second, dict)
        and (second.get("checks") or {}).get("deciding") == "passed"
        and (latex := _text(second.get("quadratic_form_latex"))) is not None
    ):
        out.append(("second", latex))
    return out


def _conditions_established(items: object) -> bool:
    if not isinstance(items, list):
        return True
    return all(isinstance(c, dict) and c.get("established") is True for c in items)


def statistics(data: dict[str, Any]) -> Listed:
    """`estimate:<parameter>`, `moment:<name>` and `expr:<name>` of an accepted `statistical_model` card.

    The estimate row is `\\hat{p} = value`, exactly as the panel draws it; a moment row is the value the panel
    draws next to its label (an expression); an expression row is the log-likelihood or a score. Nothing is
    listed when the result holds given a premise (`evidence == "conditional"` or any premise: the iid
    assumption, a prior) or rests on a condition that was assumed and not established, and an estimate that
    exists only under a condition that was not established is left out. `fisher` is the information matrix of a
    single observation, as one formula (it passed both of its own checks: the variance of the score by
    quadrature and a zero expected score); with several observations it holds given iid and is not numbered.
    Not numbered: the steps, the posterior (every posterior holds given the prior, which is an assumption and
    is among the card's premises, so the rule above never lists it), the dropped factors, the checks, the
    conditions.
    """
    result = data.get("result") or {}
    if (
        result.get("status") != "ok"
        or result.get("evidence") == "conditional"
        or result.get("premises")
        or not _conditions_established(result.get("conditions"))
    ):
        return []
    out: Listed = []
    for e in result.get("estimates") or []:
        if not isinstance(e, dict) or not _conditions_established(e.get("admissible_if")):
            continue
        parameter, latex = _text(e.get("parameter")), _text(e.get("latex"))
        if parameter and latex:
            out.append((f"estimate:{parameter}", rf"\hat{{{name_latex(parameter)}}} = {latex}"))
    for key, prefix in (("moments_latex", "moment"), ("expressions_latex", "expr")):
        found = result.get(key)
        for name, latex in found.items() if isinstance(found, dict) else []:
            if (text := _text(latex)) is not None:
                out.append((f"{prefix}:{name}", text))
    if (matrix := _matrix_latex(result.get("matrix_latex"))) is not None:
        out.append(("fisher", matrix))
    return out


def _matrix_latex(rows: object) -> str | None:
    """The Fisher information matrix as the statistics panel draws it (`matrixLatex`): one `pmatrix`.

    Only a non-empty rectangular list of non-empty strings; anything else is not a matrix the card draws.
    """
    if not isinstance(rows, list) or not rows:
        return None
    if not all(isinstance(r, list) and r and len(r) == len(rows[0]) for r in rows):
        return None
    if not all(isinstance(c, str) and c.strip() for r in rows for c in r):
        return None
    body = " \\\\ ".join(" & ".join(r) for r in rows)
    return f"\\begin{{pmatrix}}{body}\\end{{pmatrix}}"


def canonical_modes(items: object) -> list[dict[str, Any]]:
    """The normal modes of an expansion in one canonical order, whatever order the solver found them in.

    `sympy.roots` returns the roots of the frequency polynomial in a dictionary order that depends on the
    version and the input, so a position in that list is not a name. Sorted by the text of omega^2, then of
    the eigenvector and of omega (plain code-point comparison, which the panel repeats exactly in
    `expansion-modes.ts`); the sort is stable, so equal modes keep their order.
    """
    found = [m for m in items if isinstance(m, dict)] if isinstance(items, list) else []
    return sorted(
        found,
        key=lambda m: (
            str(m.get("omega_squared_latex") or ""),
            str(m.get("eigenvector_latex") or ""),
            str(m.get("omega_latex") or ""),
        ),
    )


def expansion(data: dict[str, Any]) -> Listed:
    """The verified rows of an accepted `expand_motion` card.

    `equil:<name>` (`name = value`, once the equilibrium was shown to be one), `eq:<coordinate>` (an expansion
    that passed its own series check; the equation carries `+ O(...)`, a truncation), `char` (the characteristic
    polynomial of the verified linear part), `mode:<i>:omega2` / `mode:<i>:omega` (a root that was checked by
    substitution; `omega` only when omega^2 is provably positive) and `mode:<i>:vector` (`v = ...`, only when the
    engine's own residual check `(A + omega^2 I) v = 0` passed, which is the only case in which it sends one). `i`
    is the position in the canonical order of the modes (`canonical_modes`), because the solver's order
    (`sympy.roots`) is not canonical. A vector is determined only up to a scale: the number names the formula as
    drawn, not a normalisation. Not numbered: the individual terms, the matrices `A` and `B` (drawn inside one
    combined formula), unverified equations or modes, and notes.
    """
    report = data.get("report") or {}
    if report.get("status") != "ok":
        return []
    out: Listed = []
    if (report.get("equilibrium_verification") or {}).get("passed") is True:
        for e in report.get("equilibrium") or []:
            name = _text(e.get("name")) if isinstance(e, dict) else None
            latex = _text(e.get("latex")) if isinstance(e, dict) else None
            if name and latex:
                shown = _text(e.get("name_latex")) or name
                out.append((f"equil:{name}", f"{shown} = {latex}"))
    equations = [e for e in report.get("equations") or [] if isinstance(e, dict)]
    verified = [e for e in equations if (e.get("verification") or {}).get("passed") is True]
    for e in verified:
        coordinate, latex = _text(e.get("coordinate")), _text(e.get("full_latex"))
        if coordinate and latex:
            out.append((f"eq:{coordinate}", latex))
    if (char := _text(report.get("characteristic_latex"))) is not None and len(verified) == len(
        equations
    ):
        out.append(("char", char))
    for i, m in enumerate(canonical_modes(report.get("modes"))):
        if m.get("verified") is not True:
            continue
        if (omega2 := _text(m.get("omega_squared_latex"))) is not None:
            out.append((f"mode:{i}:omega2", omega2))
        if (omega := _text(m.get("omega_latex"))) is not None:
            out.append((f"mode:{i}:omega", omega))
        if (vector := _text(m.get("eigenvector_latex"))) is not None:
            out.append((f"mode:{i}:vector", f"v = {vector}"))
    return out


# ----------------------------------------------------------------------------------------------------------
# statmech, perturbation, dimensional (#133, part 2 and 3, PR 3 of 3)

# How the statistical-mechanics panel writes a base quantity: ONE closed table, `packages/ui/fixtures/statmech-symbols.json`,
# read by the panel (`statmech-panel.tsx`, `quantitySymbol`) and pinned here by a test. A name not in it (an
# occupation list, a cumulant of unstated order, a quantity of the approximation or numeric kinds, anything the user
# labelled) is drawn as the engine named it and is not numbered; so is any quantity with `variable` set, which is a
# user-labelled partial derivative whose label may equal a table name.
STATMECH_SYMBOL: dict[str, str] = {
    "Z": "Z",
    "lnZ": r"\ln Z",
    "F": "F",
    "U": "U",
    "S": "S",
    "Var_E": r"\operatorname{Var}(E)",
    "Xi": r"\Xi",
    "lnXi": r"\ln \Xi",
    "Omega": r"\Omega",
    "N": r"\langle N \rangle",
    "Var_N": r"\operatorname{Var}(N)",
    "mean_E": r"\langle E \rangle",
    "variance_E": r"\operatorname{Var}(E)",
    "mean_N": r"\langle N \rangle",
    "variance_N": r"\operatorname{Var}(N)",
}


def statmech_key(names: list[str], i: int) -> str:
    """The key of quantity i: `quantity:<name>`, and `quantity:<name>:<k>` for its k-th occurrence (k >= 2).

    A request may ask for the same base quantity twice; the key stays unique and stable, and the panel computes
    it the same way (`quantityKey` in `statmech-panel.tsx`).
    """
    name = names[i]
    k = names[: i + 1].count(name)
    return f"quantity:{name}" if k == 1 else f"quantity:{name}:{k}"


def statmech(data: dict[str, Any]) -> Listed:
    """`quantity:<name>` of an accepted `statistical_mechanics` card: `SYMBOL = formula`, exact results only.

    Only a quantity of the closed symbol table with no `variable` (a user-labelled partial derivative never is
    one). Not numbered: unevaluated representations, numerical values and asymptotic results (`result_kind`), a
    quantity one of whose independent checks disagreed, the `levels` step (the user's own input, quoted), the
    other steps, the table, the error tables, the methods and the notes. A result that is not `ok` (a
    convention still to be confirmed, a refusal) lists nothing.
    """
    result = data.get("result") or {}
    if result.get("status") != "ok":
        return []
    quantities = [q for q in result.get("quantities") or [] if isinstance(q, dict)]
    names = [str(q.get("name")) for q in quantities]
    out: Listed = []
    for i, q in enumerate(quantities):
        symbol = STATMECH_SYMBOL.get(names[i]) if q.get("variable") is None else None
        latex = _text(q.get("latex"))
        checks = [c for c in q.get("checks") or [] if isinstance(c, dict)]
        if (
            symbol is None
            or latex is None
            or q.get("result_kind", "exact") != "exact"
            or any(c.get("passed") is False for c in checks)
        ):
            continue
        out.append((statmech_key(names, i), f"{symbol} = {latex}"))
    return out


_SOLUTION_HEAD = "u(t) = "
_SOLUTION_TAIL = r"\omega t = \tau"
_SOLUTION_SEPARATOR = r",\quad "


def split_solution(latex: str) -> tuple[str, str] | None:
    """`(relation, definition)` of a perturbation `solution_latex`, or None when it is not exactly that shape.

    The engine writes `u(t) = <series> + O\\left(eps^{n}\\right),\\quad \\omega t = \\tau`: two relations. The split
    is closed: one `,\\quad ` outside every brace and `\\left ... \\right` pair, a first part that starts with
    `u(t) = ` and ends with an `O\\left(...\\right)` remainder, and a second part that is exactly `\\omega t = \\tau`.
    Anything else is not split (and so not numbered). The panel repeats this rule
    (`apps/web/src/lib/perturbation-solution.ts`); both are pinned by `perturbation-solution-split.json`.
    """
    depth = 0
    cuts: list[int] = []
    i = 0
    while i < len(latex):
        if latex.startswith("\\left", i):
            depth += 1
            i += len("\\left")
            continue
        if latex.startswith("\\right", i):
            depth -= 1
            i += len("\\right")
            continue
        ch = latex[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif depth == 0 and latex.startswith(_SOLUTION_SEPARATOR, i):
            cuts.append(i)
        if depth < 0:
            return None
        i += 1
    if depth != 0 or len(cuts) != 1:
        return None
    head = latex[: cuts[0]]
    tail = latex[cuts[0] + len(_SOLUTION_SEPARATOR) :]
    if (
        tail != _SOLUTION_TAIL
        or not head.startswith(_SOLUTION_HEAD)
        or not head.endswith(r"\right)")
        or " + O\\left(" not in head
        or head.count(" + O\\left(") != 1
        or len(head) == len(_SOLUTION_HEAD)
    ):
        return None
    return head, tail


def perturbation(data: dict[str, Any]) -> Listed:
    """`omega0`, `omega`, `ratio` and `solution` of an accepted `perturbation_solve` card, once its checks passed.

    `solution` is the first relation of `solution_latex` only (`u(t) = ... + O(...)`, see `split_solution`): the
    residual and `check_series` checks cover it. The second, `\\omega t = \\tau`, is the card's notation for the
    strained time and no check covers it, so it is shown unnumbered. A `solution_latex` that does not split
    exactly is not numbered. Not numbered either: each order's shape and frequency, and the comparison with the
    numerical run. `omega` is a truncated series without its remainder and was numbered before this reader
    knew the solution; it keeps its number (a name, not a verification).
    """
    report = data.get("report") or {}
    if report.get("status") != "ok" or (report.get("verification") or {}).get("passed") is not True:
        return []
    out: Listed = []
    for key, field in (
        ("omega0", "linear_frequency_latex"),
        ("omega", "frequency_latex"),
        ("ratio", "period_ratio_latex"),
    ):
        if (latex := _text(report.get(field))) is not None:
            out.append((key, latex))
    solution = _text(report.get("solution_latex"))
    if solution is not None and (parts := split_solution(solution)) is not None:
        out.append(("solution", parts[0]))
    return out


def dimensional(data: dict[str, Any]) -> Listed:
    """`group:<i>`: group i as the card draws it (`label = formula`), once rebuilt and found dimensionless.

    Keyed by position (the labels are index based). Not numbered: the quantities and their dimensions (the
    user's own reading), the relation (it holds only up to an undetermined function of the groups, which a
    number would not carry), the suggestions, the tests and the notes. A card that was not accepted lists nothing.
    """
    report = data.get("report") or {}
    if report.get("status") != "ok":
        return []
    out: Listed = []
    for i, g in enumerate(report.get("groups") or []):
        if not isinstance(g, dict) or g.get("dimensionless_verified") is not True:
            continue
        label, latex = _text(g.get("label")), _text(g.get("latex"))
        if label and latex:
            out.append((f"group:{i}", f"{label} = {latex}"))
    return out


# ----------------------------------------------------------------------------------------------------------
# matrix list rows (#166)


def _matrix_accepted(data: dict[str, Any]) -> dict[str, Any] | None:
    """The `result` of a matrix card when it is an accepted, evaluated one: the same gate as the whole-card number.

    Status ok, `evaluated`, and a `result_kind` that is not an equation, a representation or a numerical
    approximation (`_NOT_EVALUATED_KINDS`): those are never presented as an evaluated formula, so none of their
    rows is numbered either.
    """
    inner = data.get("result")
    if (
        not isinstance(inner, dict)
        or inner.get("status") != "ok"
        or inner.get("evaluated") is not True
        or inner.get("result_kind") in _NOT_EVALUATED_KINDS
    ):
        return None
    return inner


def matrix(data: dict[str, Any]) -> Listed:
    """`step:i` (the formula step i produced) and `alt:j` (an alternative convention) of an accepted matrix card.

    The result itself keeps its whole-card number (`result_formula`); a row whose formula is the result's gets
    that number back (one formula, one number). Both lists are in the order the engine produced them (a
    derivation is a sequence and the alternatives are listed by convention), so a position is a stable key, and
    the key is the index in the card's own list (a step that produced no formula leaves a gap, as the panel
    counts). An alternative is a separately checked entrywise form of a derivative, a labelled convention, not
    the result. Numbered rows are names only: only the whole result was verified, an intermediate step was not,
    so `@eN` of a row carries no verification (`Equation.part_of`). Not numbered: the formula a step started
    from (the input as read, or the previous step's own row), a step whose formula is a relation (the Sylvester
    equation: its `X` is defined only by the card's words), the equation and the entries, and everything of a
    result that is an equation, a representation or a numerical approximation or that was not accepted.
    """
    inner = _matrix_accepted(data)
    if inner is None:
        return []
    out: Listed = []
    steps = inner.get("steps")
    for i, step in enumerate(steps if isinstance(steps, list) else []):
        latex = _text(step.get("after_latex")) if isinstance(step, dict) else None
        # A relation (`A X - X B = ...`, the Sylvester equation of an integral) names a symbol that only the
        # card's own words define (`X` is the integral): a number would carry the formula without that sentence.
        if latex and "=" not in latex:
            out.append((f"step:{i}", latex))
    derivative = inner.get("derivative")
    alternatives = derivative.get("alternatives") if isinstance(derivative, dict) else None
    out += _rows(alternatives, "latex", "alt")
    return out


# ----------------------------------------------------------------------------------------------------------
# simplify chain steps (#166)


def simplify(data: dict[str, Any]) -> Listed:
    """`step:i` of an accepted `simplify` card: the equation `before = after` the step's own check passed.

    Every step of a chain is checked on its own (`TransformStep.verification`: the numeric comparison of the
    step's before and after forms, run by the engine's sampler and symbolic zero test, which share no code with the
    matcher that proposed the move) and the chain is re-verified end to end; a step is numbered only when its
    check ran and passed. The row is the equation the panel draws for the step (`before = after`, one formula, kind
    `equation`), in the order the engine produced the chain, keyed by position. A SymPy move is as checked as a
    handbook rule. Numbered rows are names: the check is numerical evidence (`symbolic_zero` is reported
    separately on the card and never upgrades the wording), the restrictions of the whole result do not travel
    with `@eN` of a row, so the result carries `unverified_inputs` like every list row. Not numbered: the card's
    other forms (`alternatives`, a collapsed list that is not drawn as rows), a step whose check did not pass or
    did not run, and everything of a card whose status is not `ok`. The whole result keeps its own number
    (`result_latex`); a step whose equation is the result's is a different formula (an equation) and gets its own.
    """
    if data.get("status") != "ok":
        return []
    steps = data.get("steps")
    out: Listed = []
    for i, step in enumerate(steps if isinstance(steps, list) else []):
        if not isinstance(step, dict):
            continue
        check = step.get("verification")
        if not isinstance(check, dict) or check.get("passed") is not True:
            continue
        before, after = _text(step.get("before_latex")), _text(step.get("after_latex"))
        if before and after:
            out.append((f"step:{i}", f"{before} = {after}"))
    return out


# ----------------------------------------------------------------------------------------------------------
# vector calculus rows (#166)

# Actions whose result is never numbered, with the reason (the reader checks `action`; a test pins this table).
VECTOR_NOT_NUMBERED = {
    "jacobian": "a matrix of the map to Cartesian coordinates, not an evaluated expression of the user's field",
    "substitute": "holds only given the user's premises",
    "apply": "an implication from an equation the user supplied",
    "em_derive": "rests on premises the user adopted; Minerva never checks them",
    "em_compute": "numerical electrostatics: an approximation checked against a second method",
    "integrate": "a value, often a numerical approximation, and theorems whose hypotheses are only scanned",
    "identities": "a list of identities that could be applied, none of them applied",
}


def _vector_verified(result: dict[str, Any]) -> bool:
    """The card's own check accepted the result: status ok, no premise, and no leg of the check failed."""
    if result.get("status") != "ok" or result.get("evidence") == "conditional":
        return False
    if result.get("given") or result.get("needs_premises") or result.get("blocked"):
        return False
    if result.get("integral") or result.get("em"):
        return False
    verification = result.get("verification")
    if not isinstance(verification, dict):
        return False
    return (
        verification.get("numeric_passed") is True
        and verification.get("symbolic_zero") is not False
    )


def _vector_step(step: object) -> str | None:
    """The formula of one residual-checked, premise-free step, or None."""
    if not isinstance(step, dict) or step.get("evidence") != "identity" or step.get("given"):
        return None
    verification = step.get("verification")
    if isinstance(verification, dict) and (
        verification.get("numeric_passed") is False or verification.get("symbolic_zero") is False
    ):
        return None
    return _text(step.get("after_latex"))


def vector(data: dict[str, Any]) -> Listed:
    """`result`, `component:i`, `step:i` or `distributional` of an accepted, premise-free `vector` card.

    Only what the panel draws as a formula, and only when the card's own check accepted it with no premise:
    `evaluate` and `convert` (the scalar result, or each component as the row shows it, `basis:\\quad value`,
    in the order the coordinate system fixes), the steps of `apply_identity` and `derive` when the whole result
    is an identity (every step residual-checked, no premise), and the `distributional` point-source result
    when `proof_status` is `derived` and every pairing and flux agreed (the delta statement only; the classical
    value and the user's own statement stay unnumbered). Never numbered (`VECTOR_NOT_NUMBERED`): anything that
    rests on a premise, a blocked or unaccepted result, integrals and theorems, electromagnetic derivations
    and electrostatics, the Jacobian, a list of applicable identities, `restrictions_latex`, and the classical
    value of a field with a point source (`singular_hint`: it is 0 away from the source and a number would
    present it as the divergence). A number is a name, not a verification: rows are quoted, never `known_latex`.
    """
    result = data.get("result")
    if not isinstance(result, dict) or not _vector_verified(result):
        return []
    action = result.get("action")
    if action in VECTOR_NOT_NUMBERED:
        return []
    out: Listed = []
    if action == "distributional":
        singular = result.get("singular")
        if not isinstance(singular, dict) or singular.get("proof_status") != "derived":
            return []
        checks = [*(singular.get("weak_checks") or []), *(singular.get("flux_checks") or [])]
        if not checks or not all(isinstance(c, dict) and c.get("passed") is True for c in checks):
            return []
        latex = _text(singular.get("distributional_latex"))
        return [("distributional", latex)] if latex else []
    if action in ("evaluate", "convert"):
        if result.get("singular_hint") or result.get("singular"):
            return []
        kind = result.get("result_kind")
        if kind == "scalar":
            if (latex := _text(result.get("result_latex"))) is not None:
                out.append(("result", latex))
        elif kind == "vector":
            components = result.get("components_latex")
            basis = result.get("basis_latex")
            basis = basis if isinstance(basis, list) else []
            for i, c in enumerate(components if isinstance(components, list) else []):
                if _text(c) is not None:
                    b = basis[i] if i < len(basis) and isinstance(basis[i], str) else ""
                    out.append((f"component:{i}", f"{b}:\\quad {c}"))
        return out
    if action in ("apply_identity", "derive") and result.get("evidence") == "identity":
        for i, step in enumerate(result.get("steps") or []):
            if (latex := _vector_step(step)) is not None:
                out.append((f"step:{i}", latex))
    return out


# Card types whose listed equations are numbered one by one.
LIST_READERS: dict[str, Callable[[dict[str, Any]], Listed]] = {
    "geometry": geometry,
    "lagrangian": lagrangian,
    "invariants": invariants,
    "discovery": discovery,
    "solve": solve,
    "ode_solution": ode_solution,
    "ode_analysis": ode_analysis,
    "variation": variation,
    "statistics": statistics,
    "expansion": expansion,
    "statmech": statmech,
    "perturbation": perturbation,
    "dimensional": dimensional,
    "matrix": matrix,
    "simplify": simplify,
    "vector": vector,
}

# One primary result (`result_latex`, numbered as a whole) or no formula of its own to refer to.
SINGLE_RESULT = frozenset(
    {
        "identity",
        "differentiate",
        "series",
        "integrate",
        "telescoping",
        "sums",
        "expectation",
        "distribution",
        "approximation",
        # `tensor`: one canonical form, numbered as a whole through its top-level `result_latex`.
        "tensor",
        "step",
        "identities",
        "entries",
        "entry",
        "units",
        "note",
        "error",
    }
)

# Cards that do list several formulas and are NOT numbered one by one yet. Listing a type here is a
# statement of what is missing, not a claim that it has no list: move it to LIST_READERS with a reader.
LIST_NOT_YET_NUMBERED: frozenset[str] = frozenset()

# Card types that are deliberately not numbered, each with the reason a test pins. Not a gap: a number would
# present something that is not a displayed equation, or a numerical approximation, as a result to refer to.
EXCLUDED_WITH_REASON: dict[str, str] = {
    "ode_numeric": (
        "a numerical trajectory on a stated interval: it shows values and checks, no equation to refer to"
    ),
    "motion": (
        "a numerical solution of the user's own Lagrangian; the only formula it displays is that input, quoted"
    ),
}


def listed_equations(card_type: str, data: dict[str, Any]) -> Listed:
    """The card's listed equations in display order; empty for a card that is not list-numbered."""
    reader = LIST_READERS.get(card_type)
    return reader(data) if reader else []
