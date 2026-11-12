"""Per-user saved calculation history."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Response, status
from sqlalchemy import delete, select, update

from .db import Calculation, Derivation
from .deps import DbDep, UserDep
from .schemas import HistoryItem

router = APIRouter(prefix="/v1/history", tags=["history"])


def _item(row: Calculation) -> HistoryItem:
    return HistoryItem(
        id=row.id,
        operation=row.operation,
        input=row.input_text,
        format=row.input_format,
        assumptions=row.assumptions,
        status=row.status,
        result=row.result,
        created_at=row.created_at,
        derivation_id=row.derivation_id,
        parent_id=row.parent_id,
    )


@router.get("", response_model=list[HistoryItem])
def list_history(
    user: UserDep, db: DbDep, limit: Annotated[int, Query(ge=1, le=200)] = 50
) -> list[HistoryItem]:
    rows = db.scalars(
        select(Calculation)
        .where(Calculation.user_id == user.id)
        .order_by(Calculation.created_at.desc())
        .limit(limit)
    )
    return [_item(r) for r in rows]


@router.get("/{calculation_id}", response_model=HistoryItem)
def get_history_item(calculation_id: uuid.UUID, user: UserDep, db: DbDep) -> HistoryItem:
    row = db.scalar(
        select(Calculation).where(Calculation.id == calculation_id, Calculation.user_id == user.id)
    )
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")
    return _item(row)


@router.delete("/{calculation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_history_item(calculation_id: uuid.UUID, user: UserDep, db: DbDep) -> Response:
    # Steps that continued this one keep their place but lose the link (SQLite does not cascade).
    db.execute(
        update(Calculation)
        .where(Calculation.parent_id == calculation_id, Calculation.user_id == user.id)
        .values(parent_id=None)
    )
    result = db.execute(
        delete(Calculation).where(Calculation.id == calculation_id, Calculation.user_id == user.id)
    )
    db.commit()
    if result.rowcount == 0:  # type: ignore[attr-defined]
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
def clear_history(user: UserDep, db: DbDep) -> Response:
    db.execute(delete(Calculation).where(Calculation.user_id == user.id))
    db.execute(delete(Derivation).where(Derivation.user_id == user.id))
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
