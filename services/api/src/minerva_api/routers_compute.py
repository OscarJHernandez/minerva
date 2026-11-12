"""Computation endpoints (authenticated). Each runs inside the engine sandbox."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, status
from minerva_engine import (
    DifferentiateResult,
    ExpressionInput,
    IntegrateResult,
    SeriesResult,
    TensorInput,
    TensorResult,
)
from minerva_knowledge.identities import (
    FindIdentitiesResponse,
    TransformResponse,
    find_identities,
    simplify_expression,
)

from .db import Calculation
from .deps import DbDep, EngineDep, KbDep, LimiterDep, SettingsDep, UserDep
from .logging import get_logger
from .schemas import (
    DifferentiateRequest,
    IdentitiesRequest,
    IntegrateRequest,
    SeriesRequest,
    SimplifyRequest,
)

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


class IdentitiesResponse(FindIdentitiesResponse):
    calculation_id: uuid.UUID | None = None


def _guard(user: UserDep, limiter: LimiterDep, settings: SettingsDep, body_len: int) -> None:
    if body_len > settings.max_input_chars:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Input is too long.")
    if not limiter.allow(f"compute:{user.id}", settings.compute_rate_per_minute):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Too many requests; slow down.")


def _save(
    db: DbDep,
    user: UserDep,
    operation: str,
    req: ExpressionInput,
    status_value: str,
    result: dict[str, Any],
) -> uuid.UUID:
    row = Calculation(
        user_id=user.id,
        operation=operation,
        input_text=req.input,
        input_format=req.format,
        assumptions=req.assumptions,
        status=status_value,
        result=result,
    )
    db.add(row)
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
    result = simplify_expression(
        kb,
        engine,
        ExpressionInput(**body.model_dump(exclude={"save", "goal", "target"})),
        body.goal,
        body.target,
    )
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
    expression = ExpressionInput(**body.model_dump(exclude={"save", "variables"}))
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
        out.calculation_id = _save(
            db, user, "differentiate", expression, result.status.value, saved
        )
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
    expression = ExpressionInput(**body.model_dump(exclude={"save", "variable", "point", "order"}))
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
        out.calculation_id = _save(db, user, "series", expression, result.status.value, saved)
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
    expression = ExpressionInput(**body.model_dump(exclude={"save", "variable", "lower", "upper"}))
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
        out.calculation_id = _save(db, user, "integrate", expression, result.status.value, saved)
    return out


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
    expression = ExpressionInput(**body.model_dump(exclude={"save", "query"}))
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
            db, user, "identities", expression, result.status.value, result.model_dump(mode="json")
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
