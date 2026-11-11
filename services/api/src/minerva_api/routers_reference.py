"""Public, read-only access to the handbook (the knowledge base)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from minerva_knowledge import Entry, SearchQuery

from .deps import KbDep
from .schemas import EntryOut

router = APIRouter(prefix="/v1/reference", tags=["reference"])


def _entry_out(kb: KbDep, e: Entry) -> EntryOut:
    return EntryOut(
        id=e.id,
        title=e.title,
        kind=e.kind,
        chapter=e.chapter,
        section=e.section,
        statement_latex=kb.statement_latex(e),
        validity=e.validity,
        verification=e.verification,
        tags=e.tags,
        keywords=e.keywords,
        reference=e.reference.model_dump() if e.reference else None,
        symbols=[s.model_dump() for s in e.symbols],
        constant=e.constant.model_dump() if e.constant else None,
        notes=e.notes,
        applicable_as_rule=e.rule is not None,
        conditions=[c.model_dump() for c in e.rule.conditions] if e.rule else [],
    )


@router.get("/toc")
def toc(kb: KbDep) -> list[dict[str, object]]:
    return kb.toc()


@router.get("/entries", response_model=list[EntryOut])
def list_entries(
    kb: KbDep,
    q: Annotated[str, Query(max_length=200)] = "",
    chapter: Annotated[str | None, Query(max_length=64)] = None,
    kind: Annotated[str | None, Query(max_length=32)] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[EntryOut]:
    entries = kb.search(SearchQuery(text=q, chapter=chapter, kind=kind, limit=limit))
    return [_entry_out(kb, e) for e in entries]


@router.get("/entries/{entry_id}", response_model=EntryOut)
def get_entry(entry_id: str, kb: KbDep) -> EntryOut:
    entry = kb.get(entry_id)
    if entry is None:
        raise HTTPException(404, "No such entry")
    return _entry_out(kb, entry)
