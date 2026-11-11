"""Computation endpoints (authenticated). Each runs inside the engine sandbox."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, status
from minerva_engine import ExpressionInput, SimplifyResult, TensorInput, TensorResult
from minerva_knowledge.identities import FindIdentitiesResponse, find_identities

from .db import Calculation
from .deps import DbDep, EngineDep, KbDep, LimiterDep, SettingsDep, UserDep
from .logging import get_logger
from .schemas import IdentitiesRequest, SimplifyRequest

router = APIRouter(prefix="/v1/compute", tags=["compute"])
log = get_logger()


class SimplifyResponse(SimplifyResult):
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
    limiter: LimiterDep,
    settings: SettingsDep,
) -> SimplifyResponse:
    _guard(user, limiter, settings, len(body.input))
    result = engine.simplify(ExpressionInput(**body.model_dump(exclude={"save"})))
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
