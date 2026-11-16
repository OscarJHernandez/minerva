"""Workbench derivations: ordered steps (saved calculations) that continue one another.

A derivation holds no mathematics of its own. Every step is a calculation written by the engine
through the compute endpoints; this router only groups them and reads them back, per account.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import func, select, update

from .db import Calculation, Derivation, utcnow
from .deps import DbDep, UserDep
from .routers_history import _housekeeping, _item
from .schemas import DerivationOut, DerivationSummary, Restored
from .softdelete import within_window

router = APIRouter(
    prefix="/v1/derivations", tags=["derivations"], dependencies=[Depends(_housekeeping)]
)


def _own(db: DbDep, user: UserDep, derivation_id: uuid.UUID) -> Derivation:
    row = db.get(Derivation, derivation_id)
    if row is None or row.user_id != user.id or row.deleted_at is not None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")
    return row


@router.post("", response_model=DerivationOut, status_code=status.HTTP_201_CREATED)
def create_derivation(user: UserDep, db: DbDep) -> DerivationOut:
    now = utcnow()
    row = Derivation(user_id=user.id, created_at=now, updated_at=now)
    db.add(row)
    db.commit()
    return DerivationOut(id=row.id, updated_at=row.updated_at, steps=[])


@router.get("", response_model=list[DerivationSummary])
def list_derivations(user: UserDep, db: DbDep, limit: int = 30) -> list[DerivationSummary]:
    limit = max(1, min(limit, 100))
    rows = list(
        db.scalars(
            select(Derivation)
            .where(Derivation.user_id == user.id, Derivation.deleted_at.is_(None))
            .order_by(Derivation.updated_at.desc())
            .limit(limit)
        )
    )
    out: list[DerivationSummary] = []
    for d in rows:
        count = db.scalar(
            select(func.count())
            .select_from(Calculation)
            .where(Calculation.derivation_id == d.id, Calculation.deleted_at.is_(None))
        )
        if not count:
            continue  # a derivation whose first step never ran is not worth listing
        first = db.scalar(
            select(Calculation)
            .where(Calculation.derivation_id == d.id, Calculation.deleted_at.is_(None))
            .order_by(Calculation.created_at.asc())
            .limit(1)
        )
        out.append(
            DerivationSummary(
                id=d.id,
                updated_at=d.updated_at,
                steps=count,
                first_input=first.input_text if first else None,
                first_format=first.input_format if first else None,
            )
        )
    return out


@router.get("/{derivation_id}", response_model=DerivationOut)
def get_derivation(derivation_id: uuid.UUID, user: UserDep, db: DbDep) -> DerivationOut:
    row = _own(db, user, derivation_id)
    steps = db.scalars(
        select(Calculation)
        .where(
            Calculation.derivation_id == row.id,
            Calculation.user_id == user.id,
            Calculation.deleted_at.is_(None),
        )
        .order_by(Calculation.created_at.asc())
    )
    return DerivationOut(id=row.id, updated_at=row.updated_at, steps=[_item(s) for s in steps])


@router.delete("/{derivation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_derivation(derivation_id: uuid.UUID, user: UserDep, db: DbDep) -> Response:
    """Hide the derivation and its live steps under one stamp; `restore` undoes it within the window."""
    row = _own(db, user, derivation_id)
    now = utcnow()
    db.execute(
        update(Calculation)
        .where(Calculation.derivation_id == row.id, Calculation.deleted_at.is_(None))
        .values(deleted_at=now)
    )
    row.deleted_at = now
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{derivation_id}/restore", response_model=Restored)
def restore_derivation(derivation_id: uuid.UUID, user: UserDep, db: DbDep) -> Restored:
    row = db.get(Derivation, derivation_id)
    if row is None or row.user_id != user.id or row.deleted_at is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")
    if not within_window(row.deleted_at):
        raise HTTPException(status.HTTP_410_GONE, "The undo window has passed")
    ids = list(
        db.scalars(
            select(Calculation.id).where(
                Calculation.derivation_id == row.id, Calculation.deleted_at == row.deleted_at
            )
        )
    )
    db.execute(update(Calculation).where(Calculation.id.in_(ids)).values(deleted_at=None))
    row.deleted_at = None
    row.updated_at = utcnow()
    db.commit()
    return Restored(ids=ids)
