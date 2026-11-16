"""Soft delete with a short undo window, and the purge that makes a deletion final.

A deletion hides a step (and the steps that continued it) or a whole derivation and stamps them with
one time. A restore brings back exactly the rows that carry that stamp, only inside `UNDO_WINDOW`.
Nothing is ever re-sent by the client, so an undo cannot plant a result Minerva did not verify.
After the window the rows are deleted for good, for every account, by whichever request comes next
(`maybe_purge`) and at start-up: deleted data does not wait for its owner to return.
"""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .db import Calculation, Derivation, utcnow

UNDO_WINDOW = timedelta(minutes=10)
PURGE_EVERY_SECONDS = 60.0
_last_purge = 0.0


def aware(moment: datetime) -> datetime:
    """SQLite hands datetimes back without a zone; they were written as UTC."""
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def within_window(stamp: datetime) -> bool:
    return utcnow() - aware(stamp) <= UNDO_WINDOW


def subtree_ids(db: Session, user_id: uuid.UUID, root: Calculation) -> list[uuid.UUID]:
    """The step and every live step below it (through `parent_id`), in the same derivation."""
    if root.derivation_id is None:
        return [root.id]
    rows = db.execute(
        select(Calculation.id, Calculation.parent_id).where(
            Calculation.user_id == user_id,
            Calculation.derivation_id == root.derivation_id,
            Calculation.deleted_at.is_(None),
        )
    ).all()
    children: dict[uuid.UUID, list[uuid.UUID]] = {}
    for row_id, parent_id in rows:
        if parent_id is not None:
            children.setdefault(parent_id, []).append(row_id)
    found = [root.id]
    queue = [root.id]
    while queue:
        for child in children.get(queue.pop(), []):
            if child not in found:
                found.append(child)
                queue.append(child)
    return found


def purge_expired(db: Session) -> int:
    """Delete for good everything whose undo window has passed. Returns the rows removed."""
    cutoff = utcnow() - UNDO_WINDOW
    old_derivations = list(
        db.scalars(
            select(Derivation.id).where(
                Derivation.deleted_at.is_not(None), Derivation.deleted_at < cutoff
            )
        )
    )
    removed = 0
    if old_derivations:
        removed += db.execute(
            delete(Calculation).where(Calculation.derivation_id.in_(old_derivations))
        ).rowcount  # type: ignore[attr-defined]
        db.execute(delete(Derivation).where(Derivation.id.in_(old_derivations)))
    removed += db.execute(
        delete(Calculation).where(
            Calculation.deleted_at.is_not(None), Calculation.deleted_at < cutoff
        )
    ).rowcount  # type: ignore[attr-defined]
    db.commit()
    return int(removed)


def maybe_purge(db: Session) -> None:
    """`purge_expired` at most once a minute per process; a failure never reaches the request."""
    global _last_purge
    now = time.monotonic()
    if now - _last_purge < PURGE_EVERY_SECONDS:
        return
    _last_purge = now
    try:
        purge_expired(db)
    except Exception:  # housekeeping must not fail a user's request
        db.rollback()
