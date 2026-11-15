"""What can be shown about a parameter value from its declared assumptions.

SymPy's assumptions cannot express `0 < p < 1`, so the registry's constraints are decided here:
`True` (provably satisfied), `False` (provably violated: the model is refused) or `None` (not decided
by the declarations: the constraint is reported as a premise of the result, never silently assumed true
and never used to refuse).
"""

from __future__ import annotations

import sympy as sp

Constraint = str  # "real" | "positive" | "unit_open" | "positive_integer"


def decide(constraint: Constraint, value: sp.Expr) -> bool | None:
    if value.free_symbols and not value.is_number:
        return _symbolic(constraint, value)
    if not value.is_real:
        return False
    match constraint:
        case "real":
            return True
        case "positive":
            return bool(value > 0)
        case "unit_open":
            return bool(0 < value < 1)
        case "positive_integer":
            return bool(value.is_integer and value > 0)
    raise ValueError(f"unknown constraint {constraint}")


def _symbolic(constraint: Constraint, value: sp.Expr) -> bool | None:
    match constraint:
        case "real":
            return value.is_real if value.is_real is not None else None
        case "positive":
            if value.is_positive:
                return True
            return False if value.is_nonpositive else None
        case "unit_open":
            if value.is_nonpositive:
                return False
            return None  # an upper bound of 1 cannot be declared
        case "positive_integer":
            if value.is_integer and value.is_positive:
                return True
            return False if (value.is_integer is False or value.is_nonpositive) else None
    raise ValueError(f"unknown constraint {constraint}")


DESCRIPTIONS = {
    "real": "{name} is a real number",
    "positive": "{name} > 0",
    "unit_open": "0 < {name} < 1",
    "positive_integer": "{name} is a positive integer",
}
