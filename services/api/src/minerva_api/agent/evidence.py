"""What the engine established for a tool result, read from the result itself.

One place turns the many verification shapes of the engine (`Verification`, quadrature, series checks,
...) into a single `Evidence` record. It only reads fields the engine already computed; it never decides
anything. A result that was not verified has `accepted=False` whatever the card holds.
"""

from __future__ import annotations

from typing import Any

from .events import CheckKind, Evidence, EvidenceReason

# The kind of check a tool's result rests on, when it carries no more specific sign.
_CHECK_BY_TOOL: dict[str, CheckKind] = {
    "simplify": "numeric",
    "differentiate": "numeric",
    "integrate": "numeric",
    "series": "series_checks",
    "matrix_calculus": "numeric",
    "solve_polynomial": "symbolic_zero",
    "analyze_lagrangian": "numeric",
    "expand_motion": "series_checks",
    "perturbation_solve": "numeric",
    "search_invariants": "numeric",
    "integrate_motion": "numeric",
    "find_identities": "numeric",
    "distribution": "numeric",  # test-function pairings agree; conditions are established exactly
    "geometry": "numeric",  # closed-form and independent checks; a symbolic proof is reported as such
    "tensor_canonicalize": "structural",
    "dimensional_analysis": "dimensional",
    "check_units": "dimensional",
    "search_handbook": "lookup",
    "get_entry": "lookup",
}
_LOOKUPS = {"search_handbook", "get_entry"}
_NESTED = ("report", "result")  # cards that wrap the engine result
_REASONS: dict[str, EvidenceReason] = {
    "unchanged": "unchanged",
    "not_verified": "not_verified",
    "conditions_not_satisfied": "conditions_not_satisfied",
    "unsupported": "unsupported",
    "parse_error": "parse_error",
    "resource_limit": "resource_limit",
    "internal_error": "internal_error",
    "no_results": "no_results",
    "error": "error",
}


def _inner(data: dict[str, Any]) -> dict[str, Any]:
    for key in _NESTED:
        inner = data.get(key)
        if isinstance(inner, dict):
            return inner
    return data


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first_int(source: dict[str, Any], *names: str) -> int | None:
    for name in names:
        value = source.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def evidence_for(tool: str, status: str, card_data: dict[str, Any]) -> Evidence:
    """The evidence record of one tool result; `status` is the tool's own status."""
    data = _inner(card_data)
    verification = _dict(data.get("verification"))
    quadrature = _dict(data.get("quadrature"))
    lookup = tool in _LOOKUPS

    if status == "ok":
        reason: EvidenceReason = "lookup" if lookup else "verified"
    else:
        reason = _REASONS.get(status, "error")

    check: CheckKind = _CHECK_BY_TOOL.get(tool, "none")
    if status != "ok" and not lookup:
        check = "none" if status in ("unchanged", "unsupported", "parse_error", "error") else check
    symbolic_zero = verification.get("symbolic_zero")
    symbolic = symbolic_zero if isinstance(symbolic_zero, bool) else None
    if symbolic and not lookup:
        check = "symbolic_zero"
    elif tool == "integrate" and data.get("definite") is True:
        check = "quadrature"

    source = quadrature or verification
    checked = _first_int(source, "points_checked", "numeric_points_checked")
    agreed = _first_int(source, "points_agreed", "numeric_points_agreed")

    restrictions = data.get("restrictions_latex")
    complete = data.get("restrictions_complete")
    return Evidence(
        check=check,
        accepted=status == "ok",
        reason=reason,
        points_checked=checked,
        points_agreed=agreed,
        symbolic_zero=symbolic,
        restrictions=len(restrictions) if isinstance(restrictions, list) else 0,
        restrictions_complete=complete if isinstance(complete, bool) else None,
    )
