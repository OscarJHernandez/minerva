"""Per-user saved calculation history."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import delete, select, update

from .db import Calculation, Derivation, utcnow
from .deps import DbDep, UserDep
from .schemas import HistoryItem, Restored
from .softdelete import maybe_purge, subtree_ids, within_window


def _housekeeping(db: DbDep) -> None:
    maybe_purge(db)


router = APIRouter(prefix="/v1/history", tags=["history"], dependencies=[Depends(_housekeeping)])


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
        .where(Calculation.user_id == user.id, Calculation.deleted_at.is_(None))
        .order_by(Calculation.created_at.desc())
        .limit(limit)
    )
    return [_item(r) for r in rows]


@router.get("/{calculation_id}", response_model=HistoryItem)
def get_history_item(calculation_id: uuid.UUID, user: UserDep, db: DbDep) -> HistoryItem:
    row = db.scalar(
        select(Calculation).where(
            Calculation.id == calculation_id,
            Calculation.user_id == user.id,
            Calculation.deleted_at.is_(None),
        )
    )
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")
    return _item(row)


@router.delete("/{calculation_id}", response_model=Restored)
def delete_history_item(calculation_id: uuid.UUID, user: UserDep, db: DbDep) -> Restored:
    """Hide a step and the steps that continued it (a continuation of a removed step has nothing to
    continue from). One shared stamp lets `restore` bring back exactly this set."""
    row = db.scalar(
        select(Calculation).where(
            Calculation.id == calculation_id,
            Calculation.user_id == user.id,
            Calculation.deleted_at.is_(None),
        )
    )
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")
    ids = subtree_ids(db, user.id, row)
    now = utcnow()
    db.execute(
        update(Calculation)
        .where(Calculation.id.in_(ids), Calculation.user_id == user.id)
        .values(deleted_at=now)
    )
    if row.derivation_id is not None:
        derivation = db.get(Derivation, row.derivation_id)
        if derivation is not None:
            derivation.updated_at = now
    db.commit()
    return Restored(ids=ids)


@router.post("/{calculation_id}/restore", response_model=Restored)
def restore_history_item(calculation_id: uuid.UUID, user: UserDep, db: DbDep) -> Restored:
    """Undo a deletion made by `DELETE /v1/history/{id}` within the undo window."""
    row = db.scalar(
        select(Calculation).where(
            Calculation.id == calculation_id,
            Calculation.user_id == user.id,
            Calculation.deleted_at.is_not(None),
        )
    )
    if row is None or row.deleted_at is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")
    if not within_window(row.deleted_at):
        raise HTTPException(status.HTTP_410_GONE, "The undo window has passed")
    if row.derivation_id is not None:
        derivation = db.get(Derivation, row.derivation_id)
        if derivation is None or derivation.deleted_at is not None:
            raise HTTPException(status.HTTP_409_CONFLICT, "Its derivation was deleted")
        derivation.updated_at = utcnow()
    ids = list(
        db.scalars(
            select(Calculation.id).where(
                Calculation.user_id == user.id,
                Calculation.deleted_at == row.deleted_at,
                Calculation.derivation_id == row.derivation_id
                if row.derivation_id is not None
                else Calculation.id == row.id,
            )
        )
    )
    db.execute(
        update(Calculation)
        .where(Calculation.id.in_(ids), Calculation.user_id == user.id)
        .values(deleted_at=None)
    )
    db.commit()
    return Restored(ids=ids)


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
def clear_history(user: UserDep, db: DbDep) -> Response:
    db.execute(delete(Calculation).where(Calculation.user_id == user.id))
    db.execute(delete(Derivation).where(Derivation.user_id == user.id))
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
