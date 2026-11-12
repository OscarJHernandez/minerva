"""Find identities: retrieve candidates, then let the engine check and apply them."""

from __future__ import annotations

from minerva_engine import (
    Applicability,
    ConstantCandidate,
    Engine,
    ExpressionInput,
    RuleApplication,
    Status,
    TransformResult,
    split_equation,
)
from pydantic import BaseModel, Field

from .kb import KnowledgeBase
from .models import Entry, Reference
from .render import rhs_features, rule_features

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
    limit: int = 100,
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


MAX_CHAIN_CANDIDATES = 60


class TransformResponse(TransformResult):
    """A verified chain plus the handbook entry behind every rule it names."""

    entries: dict[str, EntrySummary] = Field(default_factory=dict)
    candidates_considered: int = 0


def chain_candidates(
    kb: KnowledgeBase, features: list[str], target: list[str], text: str, limit: int
) -> list[Entry]:
    """Rules a chain may use: structurally related to the input on either side, then related to
    what those rules can introduce. Retrieval only proposes; the engine checks every move."""
    wanted = {*features, *(f"f:{t.lower()}" for t in target)}
    first = [e for e in kb.candidate_rules(wanted, text, limit, sides="both") if e.rule]
    found = {e.id: e for e in first}
    reached: set[str] = set()
    for entry in first:
        reached |= rule_features(entry) | rhs_features(entry)
    # Only the functions a first-round rule can introduce are worth a second round: they are what
    # lets a chain continue (tan -> sin/cos, then a rule about sin and cos).
    reached = {f for f in reached if f.startswith("f:")} - wanted
    if reached:
        for entry in kb.candidate_rules(reached, "", limit, sides="both"):
            if entry.rule is not None:
                found.setdefault(entry.id, entry)
    return list(found.values())[:limit]


def transform_expression(
    kb: KnowledgeBase,
    engine: Engine,
    expression: ExpressionInput,
    goal: str = "simplify",
    target: list[str] | None = None,
    text: str = "",
    limit: int = MAX_CHAIN_CANDIDATES,
) -> TransformResponse:
    """Chains of verified rewrites toward `goal`, using the handbook and SymPy's own moves."""
    status, message, features, _ = engine.features(expression)
    if status is not Status.OK:
        return TransformResponse(status=status, message=message)
    candidates = chain_candidates(kb, features, target or [], text, limit)
    rules = [e.rule for e in candidates if e.rule is not None]
    result = engine.transform(expression, rules, goal, target)
    by_id = {e.id: e for e in candidates}
    used = {s.name for s in result.steps if s.kind == "rule"}
    used |= {s.name for c in result.alternatives for s in c.steps if s.kind == "rule"}
    used |= {b.rule_id for b in result.blocked}
    return TransformResponse(
        **result.model_dump(),
        entries={i: summarize(kb, by_id[i]) for i in sorted(used) if i in by_id},
        candidates_considered=len(rules),
    )


def simplify_expression(
    kb: KnowledgeBase,
    engine: Engine,
    expression: ExpressionInput,
    goal: str = "simplify",
    target: list[str] | None = None,
    text: str = "",
) -> TransformResponse:
    """The workbench and chat entry point: handbook chains first, SymPy's own simplification as a
    fallback when the chain search itself could not run (a hard resource stop), never silently."""
    result = transform_expression(kb, engine, expression, goal, target, text)
    if goal != "simplify" or result.status not in {Status.RESOURCE_LIMIT, Status.INTERNAL_ERROR}:
        return result
    plain = engine.simplify(expression)
    return TransformResponse(
        status=plain.status,
        goal="simplify",
        input_latex=plain.input_latex,
        result_latex=plain.result_latex,
        result_text=plain.result_text,
        verification=plain.verification,
        search_complete=False,
        message="The search through handbook identities did not finish, so this is SymPy's "
        "simplification alone, with no identities cited. " + plain.message,
        duration_ms=result.duration_ms + plain.duration_ms,
    )


def constant_candidates(kb: KnowledgeBase) -> list[ConstantCandidate]:
    """Handbook constants as candidates for a missing quantity: id, symbol (LaTeX) and SI unit.

    The symbol is the left side of the entry's statement (`c`, `k_B`, `\\hbar`). Entries whose
    statement has no usable left side are left out. Retrieval proposes; the engine decides.
    """
    out: list[ConstantCandidate] = []
    for entry in kb.entries:
        if entry.kind != "constant":
            continue
        if entry.constant is None or entry.statement_latex is None:
            continue
        sides = split_equation(entry.statement_latex)
        if sides is None or not sides[0]:
            continue
        out.append(ConstantCandidate(id=entry.id, latex=sides[0], unit=entry.constant.unit))
    return out
