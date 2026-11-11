"""Find identities: retrieve candidates, then let the engine check and apply them."""

from __future__ import annotations

from minerva_engine import (
    Applicability,
    Engine,
    ExpressionInput,
    RuleApplication,
    Status,
)
from pydantic import BaseModel, Field

from .kb import KnowledgeBase
from .models import Entry, Reference

_ORDER = {
    Applicability.APPLICABLE: 0,
    Applicability.CONDITIONS_NOT_SATISFIED: 1,
    Applicability.NOT_VERIFIED: 2,
    Applicability.NO_MATCH: 3,
}


class EntrySummary(BaseModel):
    id: str
    title: str
    kind: str
    chapter: str
    statement_latex: str
    validity: str
    verification: str
    reference: Reference | None = None


class IdentityMatch(BaseModel):
    entry: EntrySummary
    application: RuleApplication


class FindIdentitiesResponse(BaseModel):
    status: Status
    input_latex: str | None = None
    message: str = ""
    matches: list[IdentityMatch] = Field(default_factory=list)
    candidates_considered: int = 0
    duration_ms: int = 0


def summarize(kb: KnowledgeBase, entry: Entry) -> EntrySummary:
    return EntrySummary(
        id=entry.id,
        title=entry.title,
        kind=entry.kind,
        chapter=entry.chapter,
        statement_latex=kb.statement_latex(entry),
        validity=entry.validity,
        verification=entry.verification,
        reference=entry.reference,
    )


def find_identities(
    kb: KnowledgeBase,
    engine: Engine,
    expression: ExpressionInput,
    text: str = "",
    limit: int = 25,
) -> FindIdentitiesResponse:
    """Retrieval proposes candidates; only the engine decides what is applicable."""
    status, message, features, input_latex = engine.features(expression)
    if status is not Status.OK:
        return FindIdentitiesResponse(status=status, message=message)

    candidates = [e for e in kb.candidate_rules(features, text, limit) if e.rule is not None]
    if not candidates:
        return FindIdentitiesResponse(
            status=Status.OK,
            input_latex=input_latex,
            message="No identities in the knowledge base are structurally related to this input.",
        )

    result = engine.apply_rules(expression, [e.rule for e in candidates if e.rule is not None])
    if result.status is not Status.OK:
        return FindIdentitiesResponse(
            status=result.status,
            message=result.message,
            input_latex=input_latex,
            duration_ms=result.duration_ms,
        )

    by_id = {a.rule_id: a for a in result.applications}
    matches = [
        IdentityMatch(entry=summarize(kb, e), application=by_id[e.rule.id])
        for e in candidates
        if e.rule is not None and e.rule.id in by_id
    ]
    matches.sort(key=lambda m: _ORDER[m.application.applicability])
    return FindIdentitiesResponse(
        status=Status.OK,
        input_latex=input_latex,
        matches=matches,
        candidates_considered=len(candidates),
        duration_ms=result.duration_ms,
    )
