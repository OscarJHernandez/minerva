"""Tools the agent may call. Each is a pure, sandboxed computation or a read-only handbook lookup.

A tool returns two things: a `Card` (rendered by Minerva from the tool's own output, never by the
model) and a compact `for_model` text (what the model gets to read). Tools have no side effects, so a
confused or manipulated model can at worst produce a poor explanation.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from minerva_engine import (
    Engine,
    ExpressionInput,
    TensorInput,
    TensorSpec,
)
from minerva_knowledge import KnowledgeBase, SearchQuery
from minerva_knowledge.identities import find_identities, summarize
from pydantic import BaseModel, Field, ValidationError

from .events import Card
from .segments import normalize

MAX_INPUT = 2000


@dataclass
class ToolContext:
    engine: Engine
    kb: KnowledgeBase


@dataclass
class ToolOutput:
    card: Card
    for_model: str
    known_latex: list[str] = field(default_factory=list)


# --- argument models (also the JSON schema the model sees) ------------------------------------
class SimplifyArgs(BaseModel):
    latex: str = Field(
        max_length=MAX_INPUT, description="The expression in LaTeX, e.g. \\frac{x^2-1}{x-1}"
    )
    assumptions: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Properties of symbols: real, positive, negative, nonnegative, nonpositive, "
        'nonzero, integer, rational. Example: {"x": ["positive"]}',
    )


class IdentitiesArgs(SimplifyArgs):
    query: str = Field(default="", max_length=200, description="Optional extra search words")


class TensorArgs(BaseModel):
    expression: str = Field(
        max_length=MAX_INPUT, description="e.g. S(a,b)*A(-a,-b); a leading - lowers an index"
    )
    tensors: dict[str, TensorSpec] = Field(description="Rank and symmetry of every tensor used")
    dimension: int = Field(default=4, ge=1, le=64)


class SolveArgs(BaseModel):
    equation: str = Field(
        max_length=MAX_INPUT,
        description="A polynomial equation in LaTeX with one = sign, e.g. x^3+x^2+x+1=0",
    )
    variable: str | None = Field(
        default=None, max_length=32, description="The unknown, if the equation has several symbols"
    )
    assumptions: dict[str, list[str]] = Field(default_factory=dict)


class SearchArgs(BaseModel):
    query: str = Field(max_length=200, description="Words describing the formula or method")
    chapter: str | None = Field(default=None, max_length=64)
    kind: str | None = Field(default=None, max_length=32)
    limit: int = Field(default=6, ge=1, le=10)


class EntryArgs(BaseModel):
    id: str = Field(max_length=80, description="Handbook entry id, e.g. trig.pythagorean")


def _expr(args: SimplifyArgs) -> ExpressionInput:
    return ExpressionInput(input=args.latex, format="latex", assumptions=args.assumptions)


def _error(tool: str, message: str) -> ToolOutput:
    card = Card(
        id=uuid.uuid4().hex[:8],
        tool=tool,
        type="error",
        title="Could not run",
        data={"message": message},
    )
    return ToolOutput(card=card, for_model=f"error: {message}")


# --- tool implementations ----------------------------------------------------------------------
def run_simplify(args: SimplifyArgs, ctx: ToolContext) -> ToolOutput:
    result = ctx.engine.simplify(_expr(args))
    card = Card(
        id=uuid.uuid4().hex[:8],
        tool="simplify",
        type="simplify",
        title="Simplify",
        data=result.model_dump(mode="json"),
    )
    v = result.verification
    text = (
        f"status={result.status.value}; input_as_read=${result.input_latex}$; "
        f"result=${result.result_latex}$; "
        + (
            f"verified: numeric {v.numeric_points_agreed}/{v.numeric_points_checked}, "
            f"symbolic_zero={v.symbolic_zero}; "
            if v
            else ""
        )
        + result.message
    )
    known = [x for x in (result.input_latex, result.result_latex) if x]
    return ToolOutput(card, text, known)


def run_identities(args: IdentitiesArgs, ctx: ToolContext) -> ToolOutput:
    result = find_identities(ctx.kb, ctx.engine, _expr(args), text=args.query)
    card = Card(
        id=uuid.uuid4().hex[:8],
        tool="find_identities",
        type="identities",
        title="Identities",
        data=result.model_dump(mode="json"),
    )
    lines = [
        f"status={result.status.value}; input_as_read=${result.input_latex}$; {result.message}"
    ]
    known: list[str] = [result.input_latex] if result.input_latex else []
    for m in result.matches[:8]:
        a = m.application
        line = f"- {m.entry.id} ({m.entry.title}): {a.applicability.value}; statement=${m.entry.statement_latex}$"
        known.append(m.entry.statement_latex)
        if a.rewritten_latex:
            line += f"; rewritten=${a.rewritten_latex}$"
            known.append(a.rewritten_latex)
        if a.unmet_conditions:
            line += "; unmet: " + ", ".join(
                f"{c.variable} {c.assumption}" for c in a.unmet_conditions
            )
        lines.append(line)
    return ToolOutput(card, "\n".join(lines), known)


def run_tensor(args: TensorArgs, ctx: ToolContext) -> ToolOutput:
    try:
        data = TensorInput(input=args.expression, tensors=args.tensors, dimension=args.dimension)
    except ValidationError as exc:
        return _error("tensor_canonicalize", exc.errors()[0]["msg"])
    result = ctx.engine.tensor_canonicalize(data)
    card = Card(
        id=uuid.uuid4().hex[:8],
        tool="tensor_canonicalize",
        type="tensor",
        title="Tensor canonical form",
        data=result.model_dump(mode="json"),
    )
    text = f"status={result.status.value}; result={result.result_text}; {result.message}"
    return ToolOutput(card, text, [result.result_latex] if result.result_latex else [])


def run_solve(args: SolveArgs, ctx: ToolContext) -> ToolOutput:
    report = ctx.engine.solve(args.equation, args.variable, args.assumptions)
    methods = (
        ctx.kb.search(SearchQuery(features=report.features, chapter="math.methods", limit=6))
        if report.features
        else []
    )
    method_data = [summarize(ctx.kb, e).model_dump(mode="json") for e in methods]
    card = Card(
        id=uuid.uuid4().hex[:8],
        tool="solve_polynomial",
        type="solve",
        title="Solve",
        data={"report": report.model_dump(mode="json"), "methods": method_data},
    )
    lines = [f"status={report.status.value}; equation=${report.equation_latex}$; {report.message}"]
    known: list[str] = [report.equation_latex] if report.equation_latex else []
    for f in report.factors:
        exact = ", ".join(f"${r}$" for r in f.exact_roots_latex)
        lines.append(
            f"- factor ${f.factor_latex}$ (degree {f.degree}, multiplicity {f.multiplicity}): "
            f"radicals={f.radicals} ({f.radicals_reason})"
            + (f"; exact roots: {exact}" if exact else "")
        )
        known.append(f.factor_latex)
        known.extend(f.exact_roots_latex)
    if report.roots:
        shown = ", ".join(r.numeric + (" (real)" if r.real else "") for r in report.roots[:8])
        lines.append(f"numerical roots: {shown}")
    if report.verification:
        v = report.verification
        lines.append(
            f"verified: substitution={v.roots_substituted_back}, count={v.count_matches_degree}, "
            f"vieta={v.vieta_holds}"
        )
    lines.extend(f"note: {n}" for n in report.notes)
    if method_data:
        lines.append(
            "handbook methods: "
            + "; ".join(f"{m['id']} ({m['title']}, {m['verification']})" for m in method_data)
        )
        known.extend(m["statement_latex"] for m in method_data)
    return ToolOutput(card, "\n".join(lines), known)


def _entry_card_data(ctx: ToolContext, entry_ids: list[str]) -> list[dict[str, Any]]:
    out = []
    for entry_id in entry_ids:
        e = ctx.kb.get(entry_id)
        if e is not None:
            out.append(summarize(ctx.kb, e).model_dump(mode="json"))
    return out


def run_search(args: SearchArgs, ctx: ToolContext) -> ToolOutput:
    found = ctx.kb.search(
        SearchQuery(text=args.query, chapter=args.chapter, kind=args.kind, limit=args.limit)
    )
    ids = [e.id for e in found]
    entries = _entry_card_data(ctx, ids)
    card = Card(
        id=uuid.uuid4().hex[:8],
        tool="search_handbook",
        type="entries",
        title=f"Handbook: {args.query}",
        data={"query": args.query, "entries": entries},
    )
    if not entries:
        return ToolOutput(card, "no handbook entries matched; try other words", [])
    lines = [
        f"- {e['id']} ({e['title']}; verification={e['verification']}): ${e['statement_latex']}$"
        + (f" Valid: {e['validity']}" if e["validity"] else "")
        for e in entries
    ]
    return ToolOutput(card, "\n".join(lines), [e["statement_latex"] for e in entries])


def run_get_entry(args: EntryArgs, ctx: ToolContext) -> ToolOutput:
    entry = ctx.kb.get(args.id)
    if entry is None:
        return _error("get_entry", f"no handbook entry with id {args.id!r}; use search_handbook")
    data = summarize(ctx.kb, entry).model_dump(mode="json")
    data["symbols"] = [s.model_dump() for s in entry.symbols]
    data["notes"] = entry.notes
    card = Card(
        id=uuid.uuid4().hex[:8], tool="get_entry", type="entry", title=entry.title, data=data
    )
    text = (
        f"{entry.id}: {entry.title}; verification={entry.verification}; statement=${data['statement_latex']}$; "
        f"valid: {entry.validity or 'not restricted'}; notes: {entry.notes or '-'}"
    )
    return ToolOutput(card, text, [data["statement_latex"]])


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    run: Callable[[Any, ToolContext], ToolOutput]

    def schema(self) -> dict[str, Any]:
        params = self.args_model.model_json_schema()
        params.pop("title", None)
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": params},
        }


TOOLS: dict[str, Tool] = {
    t.name: t
    for t in [
        Tool(
            "search_handbook",
            "Search the handbook of verified formulas, identities, integrals, series, physics equations and constants.",
            SearchArgs,
            run_search,
        ),
        Tool(
            "get_entry",
            "Read one handbook entry by id (statement, validity conditions, how it was verified).",
            EntryArgs,
            run_get_entry,
        ),
        Tool(
            "simplify",
            "Simplify an expression with SymPy and verify the result numerically and symbolically. Reports 'unchanged' if nothing simpler exists.",
            SimplifyArgs,
            run_simplify,
        ),
        Tool(
            "solve_polynomial",
            "Solve a polynomial equation: factors it, says whether a formula in radicals exists "
            "(Galois criterion, Abel-Ruffini), gives exact roots where possible and verified numerical "
            "roots, and suggests handbook methods. Not for non-polynomial equations.",
            SolveArgs,
            run_solve,
        ),
        Tool(
            "find_identities",
            "Find handbook identities that match parts of an expression, check their validity conditions against the declared assumptions, and apply them with verification.",
            IdentitiesArgs,
            run_identities,
        ),
        Tool(
            "tensor_canonicalize",
            "Canonical form of a tensor expression using declared index symmetries (symmetric, antisymmetric, riemann). Applies monoterm symmetries only.",
            TensorArgs,
            run_tensor,
        ),
    ]
}


def schemas() -> list[dict[str, Any]]:
    return [t.schema() for t in TOOLS.values()]


def execute(name: str, arguments: dict[str, Any], ctx: ToolContext) -> ToolOutput:
    """Validate and run one tool call. Never raises: problems become an error result the model can read."""
    tool = TOOLS.get(name)
    if tool is None:
        return _error(name, f"unknown tool {name!r}; available: {', '.join(TOOLS)}")
    try:
        args = tool.args_model(**arguments)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"])
        return _error(name, f"invalid arguments ({where}): {first['msg']}")
    except TypeError:
        return _error(name, "invalid arguments")
    try:
        return tool.run(args, ctx)
    except ValidationError as exc:
        return _error(name, f"invalid arguments: {exc.errors()[0]['msg']}")


def known_formulas(outputs: list[ToolOutput]) -> set[str]:
    return {normalize(x) for o in outputs for x in o.known_latex if x}


def describe_arguments(arguments: dict[str, Any]) -> str:
    return json.dumps(arguments, ensure_ascii=False)[:400]
