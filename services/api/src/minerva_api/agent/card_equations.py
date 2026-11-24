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
    exists only under a condition that was not established is left out. Not numbered: the steps, the Fisher
    information matrix and the posterior (one joined formula), the dropped factors, the checks, the conditions.
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
    return out


def expansion(data: dict[str, Any]) -> Listed:
    """The verified rows of an accepted `expand_motion` card.

    `equil:<name>` (`name = value`, once the equilibrium was shown to be one), `eq:<coordinate>` (an expansion
    that passed its own series check; the equation carries `+ O(...)`, a truncation), `char` (the characteristic
    polynomial of the verified linear part) and `mode:<i>:omega2` / `mode:<i>:omega` (a root that was checked by
    substitution; `omega` only when omega^2 is provably positive). Not numbered: the individual terms, the
    matrices `A` and `B` (drawn inside one combined formula), the eigenvectors (the order of the modes is the
    solver's, not a canonical one, so a vector row is deferred), unverified equations or modes, and notes.
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
    for i, m in enumerate(report.get("modes") or []):
        if not isinstance(m, dict) or m.get("verified") is not True:
            continue
        if (omega2 := _text(m.get("omega_squared_latex"))) is not None:
            out.append((f"mode:{i}:omega2", omega2))
        if (omega := _text(m.get("omega_latex"))) is not None:
            out.append((f"mode:{i}:omega", omega))
    return out


# ----------------------------------------------------------------------------------------------------------
# statmech, perturbation, dimensional (#133, part 2 and 3, PR 3 of 3)

# How the statistical-mechanics panel writes a base quantity (`statmech-panel.tsx`, `SYMBOL`): a test reads the
# table from the TSX source. A quantity whose name is not here (a grand-canonical `Xi`, an occupation, a mean or a
# partial derivative with the user's own label) is drawn with its raw name, which the LaTeX reader would take
# for a product of letters, so it is not numbered.
STATMECH_SYMBOL = {
    "Z": "Z",
    "lnZ": r"\ln Z",
    "F": "F",
    "U": "U",
    "S": "S",
    "Var_E": r"\operatorname{Var}(E)",
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

    Not numbered: unevaluated representations, numerical values and asymptotic results (`result_kind`), a
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
        symbol = STATMECH_SYMBOL.get(names[i])
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


def perturbation(data: dict[str, Any]) -> Listed:
    """`omega0`, `omega` and `ratio` of an accepted `perturbation_solve` card, once its checks passed.

    Not numbered: the solution (two relations joined by `,\\quad` with an `O()` remainder: one number would
    name both, so it is deferred), each order's shape and frequency, and the comparison with the numerical run.
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
}

# One primary result (`result_latex`, numbered as a whole) or no formula of its own to refer to.
SINGLE_RESULT = frozenset(
    {
        "simplify",
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
LIST_NOT_YET_NUMBERED = frozenset(
    {
        # Deferred, with the reason: `vector` (its rows are field expressions of several actions).
        "vector",
    }
)

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
