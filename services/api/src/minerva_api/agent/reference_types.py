"""Which `@eN` a tool argument can take: a relation (`y = x^2`) or an expression (`x^2`).

`refs.ResultStore.resolve` substitutes the exact LaTeX of a numbered equation into a tool argument. Before this
module it did not look at what the equation *was*: a relation passed as the whole argument of a tool that reads
an expression failed in the LaTeX parser with "unexpected '='", and an embedded one became `\\left(x = ...\\right)`.
Now each numbered formula has a kind (`equations.relation_kind`: a top-level `=`, no relation, or undecided) and
each (tool, formula field) says what it takes. A kind that the field provably cannot read is refused with a typed
diagnostic naming the equation; **nothing is ever rewritten or split**: which side of an equation to use is the
user's decision (or the model's, in a new call that writes the expression it means).

The table is closed and tested. A field is classified only where the tool's own parser was shown to refuse a
relation (`expression_only`) or to require one (`equation_required`); everything else stays `unclassified`, with
the reason, and gets no check (current behaviour). `test_reference_types.py` fails for a tool with a formula field
that is not in the table, and re-runs every classification against the real tool.
"""

from __future__ import annotations

from typing import Literal

from minerva_engine.notation.models import InputCode, InputDiagnostic

Policy = Literal["expression_only", "equation_required", "unclassified"]

# (tool, field) -> (policy, reason). A reason is mandatory for every entry (a test pins it).
FIELD_POLICY: dict[tuple[str, str], tuple[Policy, str]] = {
    # Expression only: the parser stops at `=` ("unexpected '='" or "takes an expression, not an equation").
    ("simplify", "latex"): ("expression_only", "the parser refuses a relation"),
    ("differentiate", "latex"): ("expression_only", "the parser refuses a relation"),
    ("series", "latex"): ("expression_only", "the parser refuses a relation"),
    ("integrate", "latex"): ("expression_only", "the parser refuses a relation"),
    ("distribution", "latex"): ("expression_only", "the parser refuses a relation"),
    ("time_derivative", "latex"): ("expression_only", "the parser refuses a relation"),
    ("find_identities", "latex"): ("expression_only", "the parser refuses a relation"),
    ("sums", "latex"): ("expression_only", "the parser refuses a relation"),
    ("expectation", "latex"): ("expression_only", "the parser refuses a relation"),
    ("creative_telescoping", "latex"): ("expression_only", "the parser refuses a relation"),
    ("approximation", "latex"): ("expression_only", "the parser refuses a relation"),
    ("substitute", "latex"): (
        "expression_only",
        "this operation takes an expression, not an equation",
    ),
    ("rewrite_part", "latex"): (
        "expression_only",
        "this operation takes an expression, not an equation",
    ),
    ("variation", "integrand"): ("expression_only", "the integrand parser refuses a relation"),
    # Equation required: the tool reads `left = right` and refuses a bare expression.
    ("solve_polynomial", "equation"): ("equation_required", "needs `left = right`"),
    ("isolate_variable", "equation"): ("equation_required", "needs `left = right`"),
    ("check_identity", "latex"): ("equation_required", "needs one equation with a single `=`"),
    ("check_units", "equation"): ("equation_required", "needs an equation with an `=` sign"),
    # Not classified: no check. The reason says what was not established.
    ("matrix_calculus", "latex"): (
        "unclassified",
        "other actions read a matrix equation (Sylvester, evolution); not every action refuses one",
    ),
    ("vector_calculus", "expression"): (
        "unclassified",
        "the derive, apply and substitute actions read field equations in this argument",
    ),
    ("tensor_canonicalize", "expression"): ("unclassified", "not established for tensor input"),
    **{
        (tool, name): (
            "unclassified",
            "a leading `L =` or `H =` is read off, so a relation of that form is accepted",
        )
        for tool, name in (
            ("analyze_lagrangian", "lagrangian"),
            ("routh_reduce", "lagrangian"),
            ("time_derivative", "lagrangian"),
            ("expand_motion", "lagrangian"),
            ("perturbation_solve", "lagrangian"),
            ("find_conserved_quantities", "lagrangian"),
            ("find_conserved_quantities", "hamiltonian"),
            ("search_invariants", "lagrangian"),
            ("integrate_motion", "lagrangian"),
        )
    },
}


def policy_of(tool: str, field: str) -> Policy | None:
    """What a tool's formula field takes; None for a field this table does not know."""
    entry = FIELD_POLICY.get((tool, field))
    return entry[0] if entry else None


def refusal_diagnostic(
    tool: str, field: str, number: int, kind: Literal["equation", "expression"]
) -> tuple[InputDiagnostic, str] | None:
    """The typed refusal for `@e<number>` of this kind in this field, or None when the field accepts it.

    The diagnostic holds the reference (`@e7`), never the formula; the message is fixed text plus the equation
    number and the names of the tool and the field.
    """
    policy = policy_of(tool, field)
    if policy == "expression_only" and kind == "equation":
        code = InputCode.EQUATION_REFERENCE_NOT_EXPRESSION
        message = (
            f"equation ({number}) is a relation (it has an `=` sign), and `{tool}` reads an expression in "
            f"`{field}`. Nothing was changed. Choose which side of the equation you mean, or ask the user, and "
            "send that expression itself."
        )
        suggestions = ["send the one side of the equation you mean as the expression"]
    elif policy == "equation_required" and kind == "expression":
        code = InputCode.EXPRESSION_REFERENCE_NOT_EQUATION
        message = (
            f"equation ({number}) is an expression (it has no `=` sign), and `{tool}` needs an equation in "
            f"`{field}`. Nothing was changed. State the equation you mean, for example `left = right`."
        )
        suggestions = ["send an equation `left = right`"]
    else:
        return None
    diagnostic = InputDiagnostic(
        code=code, field=field, symbols=[f"@e{number}"], suggestions=suggestions, recoverable=False
    )
    return diagnostic, message
