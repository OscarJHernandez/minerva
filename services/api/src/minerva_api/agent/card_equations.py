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


# Card types whose listed equations are numbered one by one.
LIST_READERS: dict[str, Callable[[dict[str, Any]], Listed]] = {
    "geometry": geometry,
    "lagrangian": lagrangian,
    "invariants": invariants,
    "discovery": discovery,
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
        "matrix",
        "statistics",
        "vector",
        "statmech",
        "tensor",
        "solve",
        "expansion",
        "perturbation",
        "motion",
        "ode_analysis",
        "ode_solution",
        "ode_numeric",
        "variation",
        "dimensional",
    }
)


def listed_equations(card_type: str, data: dict[str, Any]) -> Listed:
    """The card's listed equations in display order; empty for a card that is not list-numbered."""
    reader = LIST_READERS.get(card_type)
    return reader(data) if reader else []
