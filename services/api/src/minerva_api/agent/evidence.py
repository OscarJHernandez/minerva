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
    "check_identity": "numeric",  # sampled points and targeted probes; a symbolic zero is reported as such
    "differentiate": "numeric",
    "time_derivative": "numeric",  # central difference of the input along random polynomial motions
    "integrate": "numeric",
    "series": "series_checks",
    "matrix_calculus": "numeric",
    "sums": "numeric",  # explicit instances at several sizes, never a proof for every n
    "expectation": "numeric",  # explicit finite joint distributions
    "statistical_model": "numeric",  # quadrature / direct log-likelihood / prior x likelihood
    "solve_polynomial": "symbolic_zero",
    "analyze_lagrangian": "numeric",
    "routh_reduce": "numeric",  # Euler-Lagrange residual of R on motions of L; a symbolic zero is reported apart
    "expand_motion": "series_checks",
    "perturbation_solve": "numeric",
    "search_invariants": "numeric",
    "find_conserved_quantities": "numeric",  # each constant: symbolic zero and numeric points on the flow
    "integrate_motion": "numeric",
    "ode_analyze": "numeric",  # the solved form is confirmed at sample points; the classes are not formulas
    "ode_solve_symbolic": "numeric",  # substituted back with numerical derivatives; a symbolic zero is separate
    "ode_solve_numerical": "numeric",  # independent integrator, refinement, defect, first integral
    "find_identities": "numeric",
    "distribution": "numeric",  # test-function pairings agree; conditions are established exactly
    "creative_telescoping": "symbolic_zero",  # the certificate is an exact identity; a value adds quadrature
    "vector_calculus": "numeric",
    "electromagnetism": "numeric",  # steps checked by comparing whole expressions; models are evidence only  # independent Cartesian finite differences; never a proof
    "approximation": "numeric",  # series coefficients, residual decay, quadrature or a second implementation; errors are sampled
    "variation": "numeric",  # finite differences of the action agree with the claimed variation
    "statistical_mechanics": "numeric",  # independent enumeration / quadrature; a symbolic zero is reported as such
    "geometry": "numeric",  # closed-form and independent checks; a symbolic proof is reported as such
    "substitute": "numeric",  # the replaced form reproduces the original (a congruence check, sampled)
    "rewrite_part": "numeric",  # the whole expression before and after, sampled
    "isolate_variable": "numeric",  # each solution substituted back, plus a separate completeness check
    "tensor_canonicalize": "structural",
    "dimensional_analysis": "dimensional",
    "check_units": "dimensional",
    "search_handbook": "lookup",
    "get_entry": "lookup",
    "load_tools": "lookup",  # not a result: a request for more tools
}
_LOOKUPS = {"search_handbook", "get_entry", "load_tools"}
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


def measured_verify_ms(card_data: dict[str, Any]) -> int | None:
    """The verification time the engine itself measured (`diagnostics.verify_ms` of a transformation
    step), or None when this result's operation reports none. Never derived from another duration.

    Read from the card for the trace event only: it is not part of the stored `Evidence` record (a stored
    trace and the context rebuilt from it stay exactly as before)."""
    for source in (_inner(card_data), card_data):
        diagnostics = source.get("diagnostics")
        if isinstance(diagnostics, dict) and diagnostics.get("verification_outcome") != "not_run":
            value = _first_int(diagnostics, "verify_ms")
            if value is not None:
                return value
    return None


def evidence_for(tool: str, status: str, card_data: dict[str, Any]) -> Evidence:
    """The evidence record of one tool result; `status` is the tool's own status."""
    data = _inner(card_data)
    verification = _dict(data.get("verification")) or _dict(
        _dict(data.get("first")).get("verification")
    )
    quadrature = _dict(data.get("quadrature"))
    if tool == "creative_telescoping":
        # the certificate is proved exactly; a value, if any, rests on quadrature of the original
        verification = _dict(data.get("certificate_check"))
        values = _dict(data.get("value_verification"))
        if values:
            quadrature = {
                "points_checked": values.get("quadrature_points_checked"),
                "points_agreed": values.get("quadrature_points_agreed"),
            }
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
    if tool == "creative_telescoping" and status == "ok":
        check = "quadrature" if data.get("outcome") == "evaluated" else "symbolic_zero"

    source = quadrature or verification
    checked = _first_int(
        source, "points_checked", "numeric_points_checked", "instances_checked", "models_checked"
    )
    agreed = _first_int(
        source, "points_agreed", "numeric_points_agreed", "instances_agreed", "models_agreed"
    )
    checks = [c for c in data.get("checks", []) if isinstance(c, dict)]
    if checks and checked is None and any("points_checked" in c for c in checks):
        # a statistical model: several independent checks
        checked = sum(c.get("points_checked", 0) for c in checks)
        agreed = sum(c.get("points_agreed", 0) for c in checks)

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
