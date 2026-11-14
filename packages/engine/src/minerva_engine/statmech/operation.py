"""Entry point of the ``statmech`` operation: parse, build, derive, verify, report.

A formula is shown only if every numeric check passed. Diagnostics record stage names, counts and
enumerated reasons; they never hold expressions.
"""

from __future__ import annotations

import time

from ..errors import LimitError, MinervaMathError, UnsupportedInput
from ..models import Limits, Status
from .analyze import NOT_SUPPORTED, suggest
from .canonical import (
    Canonical,
    derive,
    expectation,
    partition_steps,
    probabilities_step,
    probability_checks,
)
from .models import (
    Reason,
    Stage,
    StatCheck,
    StatMechDiagnostics,
    StatMechEvent,
    StatMechInput,
    StatMechResult,
    StatMechVerification,
    StatQuantity,
)
from .numeric import evaluate
from .oracle import MIN_POINTS, Oracle, sample_points
from .setup import Refusal, build_problem

_STATUS_FOR_REASON: dict[Reason, Status] = {
    "parse_error": Status.PARSE_ERROR,
    "invalid_degeneracy": Status.CONDITIONS_NOT_SATISFIED,
    "assumptions_insufficient": Status.CONDITIONS_NOT_SATISFIED,
    "coupling_not_linear": Status.CONDITIONS_NOT_SATISFIED,
    "ensemble_mismatch": Status.UNSUPPORTED,
    "unknown_observable": Status.UNSUPPORTED,
    "unsupported_system": Status.UNSUPPORTED,
    "verification_failed": Status.NOT_VERIFIED,
    "resource_limit": Status.RESOURCE_LIMIT,
    "divergent": Status.CONDITIONS_NOT_SATISFIED,
    "convergence_not_provable": Status.CONDITIONS_NOT_SATISFIED,
    "not_separable": Status.UNSUPPORTED,
    "convention_unconfirmed": Status.CONDITIONS_NOT_SATISFIED,
    "not_hermitian": Status.CONDITIONS_NOT_SATISFIED,
    "operators_do_not_commute": Status.CONDITIONS_NOT_SATISFIED,
    "energy_not_attainable": Status.CONDITIONS_NOT_SATISFIED,
}

# Checks whose failure means the formula is not shown. A symbolic zero that is undecided is extra
# evidence missing, not a refutation.
_DECIDING = frozenset(
    {
        "enumeration",
        "finite_difference",
        "fluctuation_response",
        "normalisation",
        "nonnegative_variance",
        "free_energy_identity",
        "spectral_decomposition",
    }
)


class _Trace:
    def __init__(self) -> None:
        self.events: list[StatMechEvent] = []
        self._t = time.monotonic()

    def mark(self, stage: Stage, status: str = "ok", reason: str | None = None) -> None:
        now = time.monotonic()
        self.events.append(
            StatMechEvent(
                stage=stage,
                status="ok" if status == "ok" else "failed",
                duration_ms=round((now - self._t) * 1000, 3),
                reason=reason,
            )
        )
        self._t = now


def _failure(
    request: StatMechInput,
    reason: Reason,
    message: str,
    trace: _Trace,
    diag: StatMechDiagnostics,
    questions: list[str] | None = None,
) -> StatMechResult:
    diag.reason = reason
    diag.events = trace.events
    return StatMechResult(
        status=_STATUS_FOR_REASON[reason],
        action=request.action,
        ensemble=request.ensemble,
        reason=reason,
        message=message,
        questions=questions or [],
        diagnostics=diag,
    )


def statmech(request: StatMechInput, limits: Limits) -> StatMechResult:
    started = time.monotonic()
    deadline = started + 0.8 * limits.timeout_seconds
    trace = _Trace()
    levels = getattr(request.system, "levels", None)
    diag = StatMechDiagnostics(
        levels=len(levels) if levels is not None else 0, parameters=len(request.parameters)
    )
    stage: Stage = "parse"
    try:
        problem = build_problem(request, limits)
        trace.mark("parse")
        stage = "build"
        notes: list[str] = list(problem.notes)
        table = None
        if request.action == "analyze":
            stage = "analyze"
            result = StatMechResult(
                status=Status.OK,
                action="analyze",
                ensemble=request.ensemble,
                state_variables=[s.name for s in problem.state_variables],
                methods=suggest(problem),
                not_supported=list(NOT_SUPPORTED),
                steps=partition_steps(problem, Canonical(problem))[:1],
                message="Methods that apply to this system, and whether Minerva implements them.",
            )
            trace.mark("analyze")
            result.diagnostics = _diagnostics(diag, trace)
            return result
        if request.ensemble == "microcanonical":
            from .micro import microcanonical

            steps, quantities, conditions = microcanonical(problem, request, deadline)
        elif request.action == "partition" and problem.kind == "occupation":
            from .grand import partition_grand

            steps, quantities, conditions = partition_grand(problem, request, deadline)
        elif request.action == "partition":
            steps, quantities, conditions = derive(
                problem,
                request.model_copy(update={"quantities": ["Z", "lnZ"], "partials": []}),
                deadline,
            )
            if problem.kind == "levels" and problem.copies_count is None:
                canon = Canonical(problem)
                steps.append(probabilities_step(canon))
                quantities.append(
                    StatQuantity(
                        name="probabilities",
                        title="Probability of each level",
                        latex=steps[-1].latex,
                        text=steps[-1].text,
                        route="P_i = g_i exp(-beta E_i) / Z",
                        checks=probability_checks(canon, Oracle(problem), sample_points(problem)),
                    )
                )
        elif request.action == "derive":
            steps, quantities, conditions = derive(problem, request, deadline)
        elif request.action == "evaluate":
            steps, quantities, conditions, table = evaluate(problem, request, deadline)
        else:
            steps, quantities, conditions = expectation(problem, request, deadline)
        trace.mark("symbolic")
        stage = "verify"
        if time.monotonic() > deadline:
            raise Refusal("resource_limit", "the derivation took too long")
        checks = [c for q in quantities for c in q.checks]
        deciding = [c for c in checks if c.name in _DECIDING]
        failed = [c for c in deciding if c.passed is not True]
        verification = StatMechVerification(
            points_checked=min((c.points for c in deciding), default=0),
            points_agreed=min((c.agreed for c in deciding), default=0),
            symbolic_zero=_symbolic(checks),
            passed=not failed and bool(deciding),
            note="numerical agreement is evidence, not a proof"
            + ("; symbolic zero found for some subjects" if _symbolic(checks) else ""),
        )
        diag.points_checked = verification.points_checked
        diag.points_agreed = verification.points_agreed
        exact_counts = request.action == "evaluate" or request.ensemble == "microcanonical"
        if failed or verification.points_checked < (1 if exact_counts else MIN_POINTS):
            trace.mark("verify", "failed", "verification_failed")
            return _failure(
                request,
                "verification_failed",
                "A result did not agree with the independent numerical evaluation, so nothing is shown.",
                trace,
                diag,
            )
        trace.mark("verify")
        notes.append("Every result above agreed with an independent numerical evaluation.")
        return StatMechResult(
            status=Status.OK,
            action=request.action,
            ensemble=request.ensemble,
            state_variables=[s.name for s in problem.state_variables],
            steps=steps,
            quantities=quantities,
            conditions=conditions,
            table=table,
            verification=verification,
            diagnostics=_diagnostics(diag, trace),
            notes=notes,
            message="Verified against an independent numerical evaluation.",
        )
    except Refusal as exc:
        trace.mark(stage, "failed", exc.reason)
        return _failure(request, exc.reason, str(exc), trace, diag, exc.questions)
    except LimitError as exc:
        trace.mark(stage, "failed", "resource_limit")
        return _failure(request, "resource_limit", str(exc), trace, diag)
    except UnsupportedInput as exc:
        trace.mark(stage, "failed", "unsupported_system")
        return _failure(request, "unsupported_system", str(exc), trace, diag)
    except MinervaMathError as exc:
        trace.mark(stage, "failed", "parse_error")
        return _failure(request, "parse_error", str(exc), trace, diag)


def _symbolic(checks: list[StatCheck]) -> bool | None:
    found = [c for c in checks if c.name == "symbolic_zero"]
    if not found:
        return None
    return True if all(c.passed for c in found) else None


def _diagnostics(diag: StatMechDiagnostics, trace: _Trace) -> StatMechDiagnostics:
    diag.events = trace.events
    return diag
