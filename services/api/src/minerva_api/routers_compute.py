"""Computation endpoints (authenticated). Each runs inside the engine sandbox."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from minerva_engine import (
    Applicability,
    DifferentiateResult,
    DiscoveryReport,
    DistributionInput,
    DistributionResult,
    ExpectationInput,
    ExpectationResult,
    ExpressionInput,
    GeometryInput,
    GeometryResult,
    IntegrateResult,
    IsolateResult,
    MatrixCalculusInput,
    MatrixCalculusResult,
    RuleCondition,
    SeriesResult,
    StatModelInput,
    StatModelResult,
    Status,
    StepResult,
    SumsInput,
    SumsResult,
    TensorInput,
    TensorResult,
    Verification,
)
from minerva_knowledge.identities import (
    EntrySummary,
    FindIdentitiesResponse,
    TransformResponse,
    find_identities,
    simplify_expression,
    summarize,
)
from pydantic import BaseModel, Field

from .db import Calculation, Derivation, utcnow
from .deps import DbDep, EngineDep, KbDep, LimiterDep, SettingsDep, UserDep
from .diagnostics import log_distribution
from .geometry_log import log_geometry
from .logging import get_logger
from .schemas import (
    ApplyIdentityRequest,
    ConservedRequest,
    DifferentiateRequest,
    DistributionRequest,
    ExpectationRequest,
    GeometryRequest,
    IdentitiesRequest,
    IntegrateRequest,
    IsolateRequest,
    Linked,
    Links,
    MatrixCalculusRequest,
    RewriteRequest,
    SeriesRequest,
    SimplifyRequest,
    StatisticsRequest,
    SubstituteRequest,
    SubtermsRequest,
    SumsRequest,
)
from .search_log import log_search_events

router = APIRouter(prefix="/v1/compute", tags=["compute"])
log = get_logger()


class SimplifyResponse(TransformResponse):
    calculation_id: uuid.UUID | None = None


class DifferentiateResponse(DifferentiateResult):
    calculation_id: uuid.UUID | None = None


class SeriesResponse(SeriesResult):
    calculation_id: uuid.UUID | None = None


class IntegrateResponse(IntegrateResult):
    calculation_id: uuid.UUID | None = None


class MatrixCalculusResponse(MatrixCalculusResult):
    calculation_id: uuid.UUID | None = None


class GeometryResponse(GeometryResult):
    calculation_id: uuid.UUID | None = None


class DistributionResponse(DistributionResult):
    calculation_id: uuid.UUID | None = None


class SumsResponse(SumsResult):
    calculation_id: uuid.UUID | None = None


class ExpectationResponse(ExpectationResult):
    calculation_id: uuid.UUID | None = None


class StatisticsResponse(StatModelResult):
    calculation_id: uuid.UUID | None = None


class IdentitiesResponse(FindIdentitiesResponse):
    calculation_id: uuid.UUID | None = None


def _guard(user: UserDep, limiter: LimiterDep, settings: SettingsDep, body_len: int) -> None:
    if body_len > settings.max_input_chars:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Input is too long.")
    if not limiter.allow(f"compute:{user.id}", settings.compute_rate_per_minute):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Too many requests; slow down.")


def _expr(body: ExpressionInput) -> ExpressionInput:
    """Only the expression fields: request extras (save, links, parameters) stay out of the engine."""
    return ExpressionInput(input=body.input, format=body.format, assumptions=body.assumptions)


def _check_links(db: DbDep, user: UserDep, body: Links) -> None:
    """A step may only join the user's own derivation and continue one of its own steps."""
    if not body.save:
        return
    if body.derivation_id is None:
        if body.parent_id is not None:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT, "A parent needs a derivation."
            )
        return
    derivation = db.get(Derivation, body.derivation_id)
    if derivation is None or derivation.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")
    if body.parent_id is not None:
        parent = db.get(Calculation, body.parent_id)
        if parent is None or parent.user_id != user.id or parent.derivation_id != derivation.id:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                "The step to continue from is not in this derivation.",
            )


def _save(
    db: DbDep,
    user: UserDep,
    operation: str,
    req: Linked,
    status_value: str,
    result: dict[str, Any],
) -> uuid.UUID:
    return _save_row(
        db, user, operation, req.input, req.format, req.assumptions, req, status_value, result
    )


def _save_row(
    db: DbDep,
    user: UserDep,
    operation: str,
    input_text: str,
    input_format: str,
    assumptions: dict[str, Any],
    links: Links,
    status_value: str,
    result: dict[str, Any],
) -> uuid.UUID:
    row = Calculation(
        user_id=user.id,
        operation=operation,
        input_text=input_text,
        input_format=input_format,
        assumptions=assumptions,
        status=status_value,
        result=result,
        derivation_id=links.derivation_id,
        parent_id=links.parent_id,
    )
    db.add(row)
    if links.derivation_id is not None:
        derivation = db.get(Derivation, links.derivation_id)
        if derivation is not None:
            derivation.updated_at = utcnow()
    db.commit()
    return row.id


@router.post("/simplify", response_model=SimplifyResponse)
def simplify(
    body: SimplifyRequest,
    user: UserDep,
    db: DbDep,
    engine: EngineDep,
    kb: KbDep,
    limiter: LimiterDep,
    settings: SettingsDep,
) -> SimplifyResponse:
    _guard(user, limiter, settings, len(body.input))
    _check_links(db, user, body)
    result = simplify_expression(kb, engine, _expr(body), body.goal, body.target)
    log.info(
        "compute",
        operation="simplify",
        status=result.status.value,
        duration_ms=result.duration_ms,
        user_id=str(user.id),
    )
    out = SimplifyResponse(**result.model_dump())
    if body.save:
        out.calculation_id = _save(
            db, user, "simplify", body, result.status.value, result.model_dump(mode="json")
        )
    return out


@router.post("/differentiate", response_model=DifferentiateResponse)
def differentiate(
    body: DifferentiateRequest,
    user: UserDep,
    db: DbDep,
    engine: EngineDep,
    limiter: LimiterDep,
    settings: SettingsDep,
) -> DifferentiateResponse:
    _guard(user, limiter, settings, len(body.input))
    _check_links(db, user, body)
    expression = _expr(body)
    result = engine.differentiate(expression, body.variables)
    log.info(
        "compute",
        operation="differentiate",
        status=result.status.value,
        duration_ms=result.duration_ms,
        user_id=str(user.id),
    )
    out = DifferentiateResponse(**result.model_dump())
    if body.save:
        saved = result.model_dump(mode="json")
        saved["variables"] = [v.model_dump() for v in body.variables]  # kept even on failure
        out.calculation_id = _save(db, user, "differentiate", body, result.status.value, saved)
    return out


@router.post("/series", response_model=SeriesResponse)
def series(
    body: SeriesRequest,
    user: UserDep,
    db: DbDep,
    engine: EngineDep,
    limiter: LimiterDep,
    settings: SettingsDep,
) -> SeriesResponse:
    _guard(user, limiter, settings, len(body.input))
    _check_links(db, user, body)
    expression = _expr(body)
    result = engine.series(expression, body.variable, body.point, body.order)
    log.info(
        "compute",
        operation="series",
        status=result.status.value,
        duration_ms=result.duration_ms,
        user_id=str(user.id),
    )
    out = SeriesResponse(**result.model_dump())
    if body.save:
        saved = result.model_dump(mode="json")
        # Kept even on failure, so history can show what was asked.
        saved.update(variable=body.variable, point_requested=body.point, order=body.order)
        out.calculation_id = _save(db, user, "series", body, result.status.value, saved)
    return out


@router.post("/integrate", response_model=IntegrateResponse)
def integrate(
    body: IntegrateRequest,
    user: UserDep,
    db: DbDep,
    engine: EngineDep,
    limiter: LimiterDep,
    settings: SettingsDep,
) -> IntegrateResponse:
    _guard(user, limiter, settings, len(body.input))
    _check_links(db, user, body)
    expression = _expr(body)
    result = engine.integrate(expression, body.variable, body.lower, body.upper)
    log.info(
        "compute",
        operation="integrate",
        status=result.status.value,
        duration_ms=result.duration_ms,
        user_id=str(user.id),
    )
    out = IntegrateResponse(**result.model_dump())
    if body.save:
        saved = result.model_dump(mode="json")
        saved["variable"] = body.variable  # kept even on failure
        out.calculation_id = _save(db, user, "integrate", body, result.status.value, saved)
    return out


def _matrix_log_fields(result: MatrixCalculusResult, trace_id: str | None) -> dict[str, Any]:
    """Counters and enums from the diagnostics: never names, entries, expressions or bounds."""
    d = result.diagnostics
    if d is None:
        return {"trace_id": trace_id}
    verify = sum(s.ms for s in d.stages if s.name.startswith("verify"))
    fields: dict[str, Any] = {
        "trace_id": d.trace_id or trace_id,
        "matrix_method": d.method,
        "result_kind": result.result_kind,
        "commutation": d.commutation,
        "verify_ms": int(verify),
        "reason": d.failure_reason,
    }
    return {k: v for k, v in fields.items() if v is not None}


@router.post("/matrix", response_model=MatrixCalculusResponse)
def matrix_calculus(
    body: MatrixCalculusRequest,
    user: UserDep,
    db: DbDep,
    engine: EngineDep,
    limiter: LimiterDep,
    settings: SettingsDep,
    request: Request,
) -> MatrixCalculusResponse:
    _guard(user, limiter, settings, len(body.input))
    _check_links(db, user, body)
    expression = ExpressionInput(input=body.input, format=body.format, assumptions=body.assumptions)
    trace_id = str(getattr(request.state, "request_id", "")) or None
    engine_request = MatrixCalculusInput(
        expression=expression,
        matrices=body.matrices,
        commuting=body.commuting,
        invertible=body.invertible,
        self_commuting=body.self_commuting,
        action=body.action,
        variable=body.variable,
        lower=body.lower,
        upper=body.upper,
        trace_id=trace_id,
    )
    result = engine.matrix_calculus(engine_request)
    log.info(
        "compute",
        operation="matrix_calculus",
        status=result.status.value,
        duration_ms=result.duration_ms,
        user_id=str(user.id),
        **_matrix_log_fields(result, trace_id),
    )
    out = MatrixCalculusResponse(**result.model_dump())
    if body.save:
        saved = result.model_dump(mode="json")
        # the declarations are kept (even on failure) so that the workbench can restore the form
        saved["request"] = engine_request.model_dump(
            mode="json", exclude={"expression", "trace_id"}
        )
        out.calculation_id = _save(db, user, "matrix_calculus", body, result.status.value, saved)
    return out


@router.post("/conserved", response_model=DiscoveryReport)
def conserved_quantities(
    body: ConservedRequest,
    user: UserDep,
    engine: EngineDep,
    limiter: LimiterDep,
    settings: SettingsDep,
) -> DiscoveryReport:
    """Adaptive search for conserved quantities. Not saved as a workbench step: the report is returned."""
    _guard(user, limiter, settings, len(body.function))
    report = engine.discover_invariants(
        body.function,
        body.coordinates,
        body.system,
        body.momenta,
        body.angles,
        body.budget,
        body.pins,
        body.time,
        body.assumptions,
    )
    log_search_events(report.events)
    log.info(
        "compute",
        operation="discover_invariants",
        status=report.status.value,
        duration_ms=report.duration_ms,
        user_id=str(user.id),
    )
    return report


def _geometry_length(body: GeometryRequest) -> int:
    cells = [c for row in body.metric for c in row]
    if body.tensor is not None:
        cells += body.tensor.components
    return max((len(c) for c in cells), default=0)


@router.post("/geometry", response_model=GeometryResponse)
def geometry(
    body: GeometryRequest,
    user: UserDep,
    db: DbDep,
    engine: EngineDep,
    limiter: LimiterDep,
    settings: SettingsDep,
) -> GeometryResponse:
    """Metric, connection, curvature, geodesics and tensor operations in explicit coordinates.

    The whole geometry is part of the request, so repeating it with other quantities reuses the same
    problem. Each quantity is returned only if its independent numerical check passed."""
    _guard(user, limiter, settings, _geometry_length(body))
    _check_links(db, user, body)
    request = GeometryInput(**body.model_dump(exclude=set(Links.model_fields)))
    result = engine.geometry(request)
    log.info(
        "compute",
        operation="geometry",
        status=result.status.value,
        duration_ms=result.duration_ms,
        user_id=str(user.id),
    )
    log_geometry(result)
    out = GeometryResponse(**result.model_dump())
    if body.save:
        saved = result.model_dump(mode="json")
        # the whole problem is kept (even on failure) so that the workbench can restore the form
        saved["request"] = request.model_dump(mode="json")
        label = f"metric in ({', '.join(request.coordinates)})"
        out.calculation_id = _save_row(
            db, user, "geometry", label, request.format, request.assumptions, body,
            result.status.value, saved,
        )  # fmt: skip
    return out


@router.post("/distribution", response_model=DistributionResponse)
def distribution(
    body: DistributionRequest,
    user: UserDep,
    db: DbDep,
    engine: EngineDep,
    limiter: LimiterDep,
    settings: SettingsDep,
) -> DistributionResponse:
    _guard(user, limiter, settings, len(body.input))
    _check_links(db, user, body)
    request = DistributionInput(
        expression=_expr(body),
        action=body.action,
        variable=body.variable,
        lower=body.lower,
        upper=body.upper,
        endpoint=body.endpoint,
        notation=body.notation,
    )
    result = engine.distribution(request)
    log.info(
        "compute",
        operation="distribution",
        status=result.status.value,
        duration_ms=result.duration_ms,
        reason=result.reason,
        user_id=str(user.id),
    )
    log_distribution(result, str(user.id))
    out = DistributionResponse(**result.model_dump())
    if body.save:
        saved = result.model_dump(mode="json")
        # Kept even on failure, so the workbench can restore the form.
        saved["request"] = request.model_dump(mode="json", exclude={"expression"})
        out.calculation_id = _save(db, user, "distribution", body, result.status.value, saved)
    return out


def _log(operation: str, result: Any, user: UserDep) -> None:
    """Statuses, durations and enumerated reasons only: never an expression, a name or a number."""
    log.info(
        "compute",
        operation=operation,
        status=result.status.value,
        duration_ms=result.duration_ms,
        reason=getattr(result, "reason", None),
        user_id=str(user.id),
    )


@router.post("/sums", response_model=SumsResponse)
def sums(
    body: SumsRequest,
    user: UserDep,
    db: DbDep,
    engine: EngineDep,
    limiter: LimiterDep,
    settings: SettingsDep,
) -> SumsResponse:
    _guard(user, limiter, settings, len(body.input))
    _check_links(db, user, body)
    request = SumsInput(
        expression=_expr(body), action=body.action, variable=body.variable, candidate=body.candidate
    )
    result = engine.sums(request)
    _log("sums", result, user)
    out = SumsResponse(**result.model_dump())
    if body.save:
        saved = result.model_dump(mode="json")
        saved["request"] = request.model_dump(mode="json", exclude={"expression"})
        out.calculation_id = _save(db, user, "sums", body, result.status.value, saved)
    return out


@router.post("/expectation", response_model=ExpectationResponse)
def expectation(
    body: ExpectationRequest,
    user: UserDep,
    db: DbDep,
    engine: EngineDep,
    limiter: LimiterDep,
    settings: SettingsDep,
) -> ExpectationResponse:
    _guard(user, limiter, settings, len(body.input))
    _check_links(db, user, body)
    request = ExpectationInput(
        expression=_expr(body),
        random=body.random,
        independent=body.independent,
        action=body.action,
        candidate=body.candidate,
    )
    result = engine.expectation(request)
    _log("expectation", result, user)
    out = ExpectationResponse(**result.model_dump())
    if body.save:
        saved = result.model_dump(mode="json")
        saved["request"] = request.model_dump(mode="json", exclude={"expression"})
        out.calculation_id = _save(db, user, "expectation", body, result.status.value, saved)
    return out


@router.post("/statistics", response_model=StatisticsResponse)
def statistics(
    body: StatisticsRequest,
    user: UserDep,
    db: DbDep,
    engine: EngineDep,
    limiter: LimiterDep,
    settings: SettingsDep,
) -> StatisticsResponse:
    _guard(user, limiter, settings, _model_length(body))
    _check_links(db, user, body)
    request = StatModelInput(**body.model_dump(exclude={"save", "derivation_id", "parent_id"}))
    result = engine.statistical_model(request)
    _log("statistical_model", result, user)
    out = StatisticsResponse(**result.model_dump())
    if body.save:
        saved = result.model_dump(mode="json")
        saved["request"] = request.model_dump(mode="json")
        label = f"{request.action} {request.family}"
        out.calculation_id = _save_row(
            db, user, "statistical_model", label, "text", request.assumptions, body,
            result.status.value, saved,
        )  # fmt: skip
    return out


def _model_length(body: StatisticsRequest) -> int:
    """The longest piece of text: every field is short, but a parameter could be an enormous expression."""
    texts = [*body.parameters.values(), *(body.data or []), *(body.candidate or {}).values()]
    if body.prior is not None:
        texts += list(body.prior.parameters.values())
    return max((len(t) for t in texts), default=0)


@router.post("/identities", response_model=IdentitiesResponse)
def identities(
    body: IdentitiesRequest,
    user: UserDep,
    db: DbDep,
    engine: EngineDep,
    kb: KbDep,
    limiter: LimiterDep,
    settings: SettingsDep,
) -> IdentitiesResponse:
    _guard(user, limiter, settings, len(body.input))
    _check_links(db, user, body)
    expression = _expr(body)
    result = find_identities(kb, engine, expression, text=body.query)
    log.info(
        "compute",
        operation="identities",
        status=result.status.value,
        duration_ms=result.duration_ms,
        user_id=str(user.id),
    )
    out = IdentitiesResponse(**result.model_dump())
    if body.save:
        out.calculation_id = _save(
            db, user, "identities", body, result.status.value, result.model_dump(mode="json")
        )
    return out


class ApplyIdentityResponse(BaseModel):
    """One handbook identity applied to an expression. The rewrite is recomputed and verified here."""

    status: Status
    input_latex: str | None = None
    rule_id: str
    entry: EntrySummary
    matched_subexpression_latex: str | None = None
    result_latex: str | None = None
    result_text: str | None = None
    unmet_conditions: list[RuleCondition] = Field(default_factory=list)
    verification: Verification | None = None
    # Points the rewrite excludes that the input, as written, did not (`x \neq 1`).
    restrictions_latex: list[str] = Field(default_factory=list)
    restrictions_complete: bool = True
    message: str = ""
    duration_ms: int = 0
    calculation_id: uuid.UUID | None = None


@router.post("/apply-identity", response_model=ApplyIdentityResponse)
def apply_identity(
    body: ApplyIdentityRequest,
    user: UserDep,
    db: DbDep,
    engine: EngineDep,
    kb: KbDep,
    limiter: LimiterDep,
    settings: SettingsDep,
) -> ApplyIdentityResponse:
    _guard(user, limiter, settings, len(body.input))
    entry = kb.get(body.rule_id)
    if entry is None or entry.rule is None:
        # `stated` and other rule-less entries are never applied.
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No applicable identity with that id")
    _check_links(db, user, body)
    result = engine.apply_rules(_expr(body), [entry.rule])
    out = ApplyIdentityResponse(
        status=result.status,
        input_latex=result.input_latex,
        rule_id=entry.id,
        entry=summarize(kb, entry),
        message=result.message,
        duration_ms=result.duration_ms,
    )
    application = next((a for a in result.applications if a.rule_id == entry.rule.id), None)
    if result.status is Status.OK and application is not None:
        out.matched_subexpression_latex = application.matched_subexpression_latex
        out.unmet_conditions = application.unmet_conditions
        out.verification = application.verification
        out.message = application.message
        out.restrictions_latex = application.restrictions_latex
        out.restrictions_complete = application.restrictions_complete
        if application.applicability is Applicability.APPLICABLE:
            out.result_latex = application.rewritten_latex
            out.result_text = application.rewritten_text
            if out.result_text is None:  # never show a rewrite without its verified text form
                out.result_latex = None
                out.status = Status.NOT_VERIFIED
        elif application.applicability is Applicability.CONDITIONS_NOT_SATISFIED:
            out.status = Status.CONDITIONS_NOT_SATISFIED
        elif application.applicability is Applicability.NOT_VERIFIED:
            out.status = Status.NOT_VERIFIED
        else:
            out.status = Status.UNCHANGED
    log.info(
        "compute",
        operation="apply_identity",
        status=out.status.value,
        duration_ms=out.duration_ms,
        user_id=str(user.id),
    )
    if body.save:
        out.calculation_id = _save(
            db, user, "apply_identity", body, out.status.value, out.model_dump(mode="json")
        )
    return out


class StepResponse(StepResult):
    """A controlled step (rewrite one part, substitute, or a list of parts)."""

    calculation_id: uuid.UUID | None = None
    entry: EntrySummary | None = None  # the handbook identity applied, if any


class IsolateResponse(IsolateResult):
    calculation_id: uuid.UUID | None = None


def _guard_total(user: UserDep, limiter: LimiterDep, settings: SettingsDep, *texts: str) -> None:
    _guard(user, limiter, settings, sum(len(t) for t in texts))


def _log_step(operation: str, user: UserDep, status_value: str, result: Any) -> None:
    """Counters, durations and codes only: `StepDiagnostics.log_fields` cannot hold an expression."""
    log.info(
        "compute",
        operation=operation,
        status=status_value,
        duration_ms=result.duration_ms,
        user_id=str(user.id),
        **result.diagnostics.log_fields(),
    )


def _saved(out: Any, request: dict[str, Any]) -> dict[str, Any]:
    """What is stored with a step: the result and the request parameters, without debug detail."""
    saved = out.model_dump(mode="json")
    saved["diagnostics"]["detail"] = []
    saved["request"] = request
    return saved  # type: ignore[no-any-return]


@router.post("/subterms", response_model=StepResponse)
def subterms(
    body: SubtermsRequest,
    user: UserDep,
    engine: EngineDep,
    kb: KbDep,
    limiter: LimiterDep,
    settings: SettingsDep,
) -> StepResponse:
    """The parts of an expression that can be chosen for a step. A lookup: never saved."""
    _guard_total(user, limiter, settings, body.input, body.find or "")
    rule = None
    if body.rule_id is not None:
        entry = kb.get(body.rule_id)
        if entry is None or entry.rule is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "No applicable identity with that id")
        rule = entry.rule
    result = engine.subterms(
        _expr(body), find=body.find, rule=rule, reverse=body.reverse, debug=settings.debug_compute
    )
    _log_step("subterms", user, result.status.value, result)
    return StepResponse(**result.model_dump())


@router.post("/rewrite", response_model=StepResponse)
def rewrite(
    body: RewriteRequest,
    user: UserDep,
    db: DbDep,
    engine: EngineDep,
    kb: KbDep,
    limiter: LimiterDep,
    settings: SettingsDep,
) -> StepResponse:
    """Apply one move (an identity, expand, factor, cancel, simplify, collect) to one chosen part."""
    _guard_total(
        user,
        limiter,
        settings,
        body.input,
        body.part or "",
        *body.by,
        body.selection.expected_latex if body.selection else "",
    )
    rule = entry = None
    if body.action == "rule":
        entry = kb.get(body.rule_id or "")
        if entry is None or entry.rule is None:
            # `stated` and other rule-less entries are never applied.
            raise HTTPException(status.HTTP_404_NOT_FOUND, "No applicable identity with that id")
        rule = entry.rule
    _check_links(db, user, body)
    result = engine.rewrite_at(
        _expr(body),
        body.action,
        selection=body.selection.model_dump() if body.selection else None,
        part=body.part,
        occurrence=body.occurrence,
        rule=rule,
        reverse=body.reverse,
        by=body.by,
        debug=settings.debug_compute,
    )
    _log_step("rewrite", user, result.status.value, result)
    out = StepResponse(**result.model_dump())
    if entry is not None:
        out.entry = summarize(kb, entry)
    if body.save:
        request = {
            "action": body.action,
            "rule_id": body.rule_id,
            "reverse": body.reverse,
            "part": body.part,
            "occurrence": body.occurrence,
            "by": body.by,
            "selection": body.selection.model_dump() if body.selection else None,
        }
        out.calculation_id = _save(
            db, user, "rewrite", body, result.status.value, _saved(out, request)
        )
    return out


@router.post("/substitute", response_model=StepResponse)
def substitute(
    body: SubstituteRequest,
    user: UserDep,
    db: DbDep,
    engine: EngineDep,
    limiter: LimiterDep,
    settings: SettingsDep,
) -> StepResponse:
    """Replace the left side of each relation by its right side. The relations are premises: a result
    that depends on them is conditional, and says so."""
    _guard_total(user, limiter, settings, body.input, *body.relations)
    _check_links(db, user, body)
    result = engine.substitute(
        _expr(body), body.relations, order=body.order, debug=settings.debug_compute
    )
    _log_step("substitute", user, result.status.value, result)
    out = StepResponse(**result.model_dump())
    if body.save:
        request = {"relations": body.relations, "order": body.order}
        out.calculation_id = _save(
            db, user, "substitute", body, result.status.value, _saved(out, request)
        )
    return out


@router.post("/isolate", response_model=IsolateResponse)
def isolate(
    body: IsolateRequest,
    user: UserDep,
    db: DbDep,
    engine: EngineDep,
    limiter: LimiterDep,
    settings: SettingsDep,
) -> IsolateResponse:
    """Solve an equation for one variable, listing every solution and the conditions it needs."""
    _guard_total(user, limiter, settings, body.input, body.variable)
    _check_links(db, user, body)
    result = engine.isolate(_expr(body), body.variable, debug=settings.debug_compute)
    _log_step("isolate", user, result.status.value, result)
    out = IsolateResponse(**result.model_dump())
    if body.save:
        out.calculation_id = _save(
            db, user, "isolate", body, result.status.value, _saved(out, {"variable": body.variable})
        )
    return out


@router.post("/tensor/canonicalize", response_model=TensorResult)
def tensor_canonicalize(
    body: TensorInput,
    user: UserDep,
    engine: EngineDep,
    limiter: LimiterDep,
    settings: SettingsDep,
) -> TensorResult:
    _guard(user, limiter, settings, len(body.input))
    result = engine.tensor_canonicalize(body)
    log.info(
        "compute",
        operation="tensor_canonicalize",
        status=result.status.value,
        duration_ms=result.duration_ms,
        user_id=str(user.id),
    )
    return result
