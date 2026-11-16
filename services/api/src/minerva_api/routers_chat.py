"""Chat with the agent: conversations, stored messages, and a streamed reply (server-sent events)."""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import copy
import json
import re
import time
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from minerva_engine import MotionRequest, VariationInput, trace_scope
from minerva_engine.distributions.models import NotationName, Reading, check_readings
from minerva_engine.statmech.models import Sweep
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .agent.budget import fit_history
from .agent.declarations import scan_declarations
from .agent.equations import (
    EquationTable,
    build_table,
    find_references,
    number_stored,
    number_user_text,
    replay_prose,
    resolve_references,
    unverified_inputs,
)
from .agent.events import AssistantMessage, Block, Card, TraceStep, TurnDebug
from .agent.llm import LLMClient, LLMError
from .agent.loop import AgentLimits, TurnInfo, matrix_names, prompt_sha, run_agent, vector_names
from .agent.openrouter import check_connection
from .agent.params import ParamError, ParamOverrides, describe, resolve
from .agent.refs import reference_note
from .agent.segments import normalize, split_segments
from .agent.statmech_tools import StatMechArgs
from .agent.tools import (
    ApproximationArgs,
    DistributionArgs,
    GeometryArgs,
    OdeAnalyzeArgs,
    OdeNumericArgs,
    OdeSymbolicArgs,
    PerturbationArgs,
    ToolContext,
    VariationArgs,
    execute,
)
from .agent.toolsets import conversation_groups
from .agent.tracing import (
    AgentCompleted,
    Sink,
    ToolCompleted,
    Tracer,
    activate,
    deactivate,
    tool_status,
)
from .agent.vision import FormulaReader, clean_candidates
from .db import ChatMessage, Conversation, User, utcnow
from .deps import DbDep, EngineDep, KbDep, LimiterDep, SettingsDep, UserDep, current_user
from .logging import get_logger
from .providers import (
    OpenRouterTest,
    ProviderChoice,
    local_model_choices,
    ollama_models,
    provider_status,
    resolve_llm,
    server_key_allowed,
)
from .settings import Settings

router = APIRouter(prefix="/v1/chat", tags=["chat"])
log = get_logger()

HISTORY_MESSAGES = 20


def _event_sink(mode: str) -> Sink | None:
    """Where trace events go as they happen: the server log, which holds no content.

    Events carry identifiers, counts, durations and fixed-vocabulary values only (agent/tracing.py), and
    the log allow-list (logging.py) drops anything else, so this cannot write an expression, a tool
    argument or result, or model text at any setting.
    """
    if mode == "off":
        return None

    def sink(record: dict[str, Any]) -> None:
        name = str(record["event"])
        if mode == "summary" and name != "agent.completed":
            return
        log.info(name, **{k: v for k, v in record.items() if k != "event" and v is not None})

    return sink


class StatusOut(BaseModel):
    enabled: bool
    mode: str
    model: str
    images: bool = False  # can formulas be read from a picture ("Add image")
    providers: dict[str, Any] = Field(default_factory=dict)
    # Parameter defaults, which fields this account may change, and whether the smallest request
    # (system prompt + core tools) fits the model's window. Absent when not signed in.
    params: dict[str, Any] | None = None
    context_ok: bool = True


class ImageIn(BaseModel):
    image: str = Field(min_length=1)  # base64 of a PNG, JPEG or WebP


class ExtractedFormula(BaseModel):
    latex: str
    status: str
    read_as: str | None = None
    message: str = ""


class ImageFormulas(BaseModel):
    formulas: list[ExtractedFormula]


class ConversationOut(BaseModel):
    id: uuid.UUID
    title: str
    updated_at: datetime


class MessageOut(BaseModel):
    id: uuid.UUID
    role: str
    content: dict[str, Any]
    created_at: datetime


class ConversationDetail(ConversationOut):
    messages: list[MessageOut]


class NewConversation(BaseModel):
    title: str | None = Field(default=None, max_length=120)


class SendMessage(BaseModel):
    content: str = Field(min_length=1, max_length=4000)
    provider: ProviderChoice | None = None  # absent: the server's default provider
    # Overrides of the model parameters for this turn (debug panel); ranges and who may change what
    # are enforced by `agent.params.resolve`.
    params: ParamOverrides | None = None
    # new: a new message. retry: answer the last user message again (the reply after it, if any, is
    # replaced; `content` is ignored). edit: replace the text of the last user message with `content` and
    # answer it again. Neither stores the user's message a second time.
    mode: Literal["new", "retry", "edit"] = "new"


class ConnectionOut(BaseModel):
    ok: bool
    message: str
    credit_remaining: float | None = None
    supports_tools: bool = False


def _llm(request: Request) -> LLMClient | None:
    return request.app.state.llm  # type: ignore[no-any-return]


def _vision(request: Request) -> FormulaReader | None:
    return request.app.state.vision  # type: ignore[no-any-return]


def _owned(db: Session, conversation_id: uuid.UUID, user_id: uuid.UUID) -> Conversation:
    row = db.scalar(
        select(Conversation).where(
            Conversation.id == conversation_id, Conversation.user_id == user_id
        )
    )
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")
    return row


@router.get("/status", response_model=StatusOut)
async def chat_status(request: Request, db: DbDep, settings: SettingsDep) -> StatusOut:
    llm = _llm(request)
    try:  # public endpoint; the per-user flag is added only when a session exists
        user: User | None = current_user(request, db, settings)
    except HTTPException:
        user = None
    providers = provider_status(settings, user)
    if settings.llm_mode == "scripted":
        providers["ollama"]["reachable"] = True
        providers["ollama"].update(local_model_choices([], settings.llm_model))
    else:
        installed = await ollama_models(request, settings)
        providers["ollama"]["reachable"] = installed is not None
        providers["ollama"].update(
            local_model_choices(installed or [], settings.llm_model if installed else "")
        )
    default_model = (
        settings.openrouter_model
        if settings.llm_mode == "openrouter"
        else (llm.name if llm else "")
    )
    return StatusOut(
        enabled=settings.llm_mode != "off",
        mode=settings.llm_mode,
        model=default_model,
        images=llm is not None and _vision(request) is not None,
        providers=providers,
        params=describe(settings, user.email) if user is not None else None,
        context_ok=_context_ok(settings),
    )


@router.post("/providers/openrouter/test", response_model=ConnectionOut)
async def test_openrouter(
    body: OpenRouterTest,
    request: Request,
    user: UserDep,
    limiter: LimiterDep,
    settings: SettingsDep,
) -> ConnectionOut:
    """Test Connection: checks the key and model without a paid completion. The key is not stored."""
    if not settings.openrouter_available:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "OpenRouter is not enabled on this server."
        )
    if not limiter.allow(f"ortest:{user.id}", 10):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Too many tests; slow down.")
    if body.api_key and body.api_key.get_secret_value():
        if not settings.openrouter_byok:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                "Personal OpenRouter keys are not enabled on this server.",
            )
        key = body.api_key.get_secret_value()
    elif server_key_allowed(settings, user):
        key = settings.openrouter_api_key.get_secret_value()
    else:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, "Add your own OpenRouter API key to test the connection."
        )
    report = await check_connection(
        key,
        body.model,
        transport=getattr(request.app.state, "openrouter_transport", None),
    )
    log.info("provider_test", provider="openrouter", status="ok" if report.ok else "failed")
    return ConnectionOut(
        ok=report.ok,
        message=report.message,
        credit_remaining=report.credit_remaining,
        supports_tools=report.supports_tools,
    )


def _image_kind(data: bytes) -> str | None:
    """The real type from the first bytes; what the client declared is not trusted."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


@router.post("/formula-image", response_model=ImageFormulas)
async def read_formula_image(
    request: Request,
    user: UserDep,
    engine: EngineDep,
    limiter: LimiterDep,
    settings: SettingsDep,
) -> ImageFormulas:
    """Transcribe formulas from a picture and show how Minerva's parser reads each one.

    Nothing is stored or logged: the picture lives in memory for this request only, and the reply is
    a proposal the user confirms before it goes into a message.
    """
    reader = _vision(request)
    if _llm(request) is None or reader is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Reading images is not set up.")
    if not limiter.allow(f"image:{user.id}", settings.image_rate_per_minute):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Too many images; slow down.")
    too_big = HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "The image is too large.")
    limit = settings.image_max_bytes * 4 // 3 + 1024  # base64 inside a small JSON envelope
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise too_big
    body = bytearray()
    async for part in request.stream():  # stop reading as soon as the cap is passed
        body += part
        if len(body) > limit:
            raise too_big
    try:
        data = base64.b64decode(ImageIn.model_validate_json(bytes(body)).image, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Unreadable image.") from exc
    if len(data) > settings.image_max_bytes:
        raise too_big
    if _image_kind(data) is None:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Use a PNG, JPEG or WebP image."
        )
    started = time.monotonic()
    outcome = "error"
    try:
        try:
            raw = await reader.read(data)
        except LLMError as exc:
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
        candidates = clean_candidates(raw, settings.max_input_chars)
        readings = await run_in_threadpool(engine.read_formulas, candidates)
        outcome = "ok"
        return ImageFormulas(
            formulas=[
                ExtractedFormula(
                    latex=latex,
                    status=reading.status.value,
                    read_as=reading.read_as,
                    message=reading.message,
                )
                for latex, reading in zip(candidates, readings, strict=True)
            ]
        )
    finally:
        log.info(
            "formula_image",
            operation="read_image",
            status=outcome,
            duration_ms=int((time.monotonic() - started) * 1000),
            user_id=str(user.id),
        )


@router.get("/conversations", response_model=list[ConversationOut])
def list_conversations(user: UserDep, db: DbDep) -> list[ConversationOut]:
    rows = db.scalars(
        select(Conversation)
        .where(Conversation.user_id == user.id)
        .order_by(Conversation.updated_at.desc())
        .limit(100)
    )
    return [ConversationOut(id=r.id, title=r.title, updated_at=r.updated_at) for r in rows]


@router.post("/conversations", response_model=ConversationOut, status_code=status.HTTP_201_CREATED)
def create_conversation(body: NewConversation, user: UserDep, db: DbDep) -> ConversationOut:
    row = Conversation(user_id=user.id, title=(body.title or "New conversation").strip()[:120])
    db.add(row)
    db.commit()
    return ConversationOut(id=row.id, title=row.title, updated_at=row.updated_at)


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
def get_conversation(conversation_id: uuid.UUID, user: UserDep, db: DbDep) -> ConversationDetail:
    row = _owned(db, conversation_id, user.id)
    messages = list(
        db.scalars(
            select(ChatMessage)
            .where(ChatMessage.conversation_id == row.id)
            .order_by(ChatMessage.created_at, ChatMessage.id)
        )
    )
    _backfill_numbers(db, row, messages)
    return ConversationDetail(
        id=row.id,
        title=row.title,
        updated_at=row.updated_at,
        messages=[
            MessageOut(id=m.id, role=m.role, content=m.content, created_at=m.created_at)
            for m in messages
        ],
    )


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(conversation_id: uuid.UUID, user: UserDep, db: DbDep) -> None:
    row = _owned(db, conversation_id, user.id)
    db.execute(delete(ChatMessage).where(ChatMessage.conversation_id == row.id))
    db.delete(row)
    db.commit()


def _run_tool_directly(
    name: str,
    arguments: dict[str, Any],
    conversation_id: uuid.UUID,
    user: Any,
    db: Any,
    engine: Any,
    kb: Any,
    limiter: Any,
    settings: Any,
    notation: dict[str, str] | None = None,
    numeric: dict[str, Any] | None = None,
    answers: dict[str, Any] | None = None,
) -> MessageOut:
    """Run one tool with the user's own numbers (from a card's form), store it as an assistant message.

    The model is not involved, so it never transcribes the numbers. The result is stored holding the card and
    the tool call, so later turns see it exactly as if the model had called the tool. Works with the chat
    model switched off.
    """
    if not limiter.allow(f"compute:{user.id}", settings.compute_rate_per_minute):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Too many requests; slow down.")
    conversation = _owned(db, conversation_id, user.id)
    trace_id = uuid.uuid4().hex
    tracer = Tracer(trace_id, _event_sink(settings.agent_events))
    activate(tracer, engine)
    started = time.monotonic()
    try:
        with trace_scope(trace_id):
            output = execute(
                name,
                arguments,
                ToolContext(
                    engine=engine,
                    kb=kb,
                    notation=dict(notation or {}),
                    numeric=dict(numeric or {}),
                    statmech_answers=dict(answers or {}),
                ),
            )
    finally:
        deactivate(tracer)
    elapsed = int((time.monotonic() - started) * 1000)
    sandbox = tracer.sandbox_since(0)
    engine_ms = sum(c.ms for c in sandbox) if sandbox else None
    category = output.error_category
    tracer.emit(
        ToolCompleted,
        None,
        tool=name,
        call_id=output.card.id,
        status=tool_status(output.status),
        duration_ms=elapsed,
        engine_ms=engine_ms,
        sandbox_calls=len(sandbox),
        error_category=category,
    )
    tracer.emit(
        AgentCompleted,
        None,
        status="error" if output.status == "error" else "ok",
        total_ms=elapsed,
        tool_calls=1,
        tool_errors=1 if output.status == "error" else 0,
        tool_ms=elapsed,
        engine_ms=engine_ms or 0,
        error_category=category,
    )
    content = AssistantMessage(
        text="",
        blocks=[Block(kind="card", card=output.card)],
        trace=[
            TraceStep(
                id=output.card.id,
                name=name,
                # The user's answer is recorded next to the call so that a replay is faithful.
                arguments={
                    **arguments,
                    **({"notation": notation} if notation else {}),
                    **({"numeric": numeric} if numeric else {}),
                    **({"answers": answers} if answers else {}),
                },
                for_model=output.for_model,
                duration_ms=elapsed,
                status=output.status,
                evidence=output.evidence,
                arguments_valid=output.arguments_valid,
                error_category=category,
                engine_ms=engine_ms,
                verified_latex=list(output.known_latex),
                quoted=dict(output.quoted_latex),
            )
        ],
        # No model took part: the debug record only says so and carries the trace id and events.
        debug=TurnDebug(
            mode="direct",
            prompt_sha=prompt_sha(),
            duration_ms=elapsed,
            trace_id=trace_id,
            stop_reason="completed",
            events=list(tracer.events),
            events_dropped=tracer.dropped,
        ),
        stop_reason="completed",
    ).model_dump(mode="json")
    row = ChatMessage(conversation_id=conversation.id, role="assistant", content=content)
    db.add(row)
    conversation.updated_at = utcnow()
    db.commit()
    log.info(
        "compute",
        operation=name,
        status=output.status,
        duration_ms=elapsed,
        user_id=str(user.id),
        trace_id=trace_id,
    )
    return MessageOut(id=row.id, role=row.role, content=row.content, created_at=row.created_at)


@router.post(
    "/conversations/{conversation_id}/tools/integrate_motion",
    response_model=MessageOut,
    status_code=status.HTTP_201_CREATED,
)
def integrate_motion_directly(
    conversation_id: uuid.UUID,
    body: MotionRequest,
    user: UserDep,
    db: DbDep,
    engine: EngineDep,
    kb: KbDep,
    limiter: LimiterDep,
    settings: SettingsDep,
) -> MessageOut:
    """Run the numerical solution with the user's own numbers (from the Lagrangian card's form)."""
    return _run_tool_directly(
        "integrate_motion",
        body.model_dump(mode="json"),
        conversation_id,
        user,
        db,
        engine,
        kb,
        limiter,
        settings,
    )


@router.post(
    "/conversations/{conversation_id}/tools/ode_solve_numerical",
    response_model=MessageOut,
    status_code=status.HTTP_201_CREATED,
)
def ode_solve_numerical_directly(
    conversation_id: uuid.UUID,
    body: OdeNumericArgs,
    user: UserDep,
    db: DbDep,
    engine: EngineDep,
    kb: KbDep,
    limiter: LimiterDep,
    settings: SettingsDep,
) -> MessageOut:
    """Integrate with the initial values, parameters and interval the user typed into a card's form."""
    return _run_tool_directly(
        "ode_solve_numerical",
        body.model_dump(mode="json"),
        conversation_id,
        user,
        db,
        engine,
        kb,
        limiter,
        settings,
    )


@router.post(
    "/conversations/{conversation_id}/tools/approximation",
    response_model=MessageOut,
    status_code=status.HTTP_201_CREATED,
)
def approximation_directly(
    conversation_id: uuid.UUID,
    body: ApproximationArgs,
    user: UserDep,
    db: DbDep,
    engine: EngineDep,
    kb: KbDep,
    limiter: LimiterDep,
    settings: SettingsDep,
) -> MessageOut:
    """Approximate with the point, interval, order, tolerance and period the user typed on the card."""
    return _run_tool_directly(
        "approximation",
        body.model_dump(mode="json"),
        conversation_id,
        user,
        db,
        engine,
        kb,
        limiter,
        settings,
    )


@router.post(
    "/conversations/{conversation_id}/tools/ode_analyze",
    response_model=MessageOut,
    status_code=status.HTTP_201_CREATED,
)
def ode_analyze_directly(
    conversation_id: uuid.UUID,
    body: OdeAnalyzeArgs,
    user: UserDep,
    db: DbDep,
    engine: EngineDep,
    kb: KbDep,
    limiter: LimiterDep,
    settings: SettingsDep,
) -> MessageOut:
    """Analyse equilibria and local stability with the parameter values and window the user typed."""
    return _run_tool_directly(
        "ode_analyze",
        body.model_dump(mode="json"),
        conversation_id,
        user,
        db,
        engine,
        kb,
        limiter,
        settings,
    )


@router.post(
    "/conversations/{conversation_id}/tools/ode_solve_symbolic",
    response_model=MessageOut,
    status_code=status.HTTP_201_CREATED,
)
def ode_solve_symbolic_directly(
    conversation_id: uuid.UUID,
    body: OdeSymbolicArgs,
    user: UserDep,
    db: DbDep,
    engine: EngineDep,
    kb: KbDep,
    limiter: LimiterDep,
    settings: SettingsDep,
) -> MessageOut:
    """Find a closed form when the user presses the button on an analysis card (with any conditions
    they typed there)."""
    return _run_tool_directly(
        "ode_solve_symbolic",
        body.model_dump(mode="json"),
        conversation_id,
        user,
        db,
        engine,
        kb,
        limiter,
        settings,
    )


@router.post(
    "/conversations/{conversation_id}/tools/perturbation_solve",
    response_model=MessageOut,
    status_code=status.HTTP_201_CREATED,
)
def perturbation_solve_directly(
    conversation_id: uuid.UUID,
    body: PerturbationArgs,
    user: UserDep,
    db: DbDep,
    engine: EngineDep,
    kb: KbDep,
    limiter: LimiterDep,
    settings: SettingsDep,
) -> MessageOut:
    """Solve again with the comparison numbers the user typed into the perturbation card's form."""
    if body.compare is None:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "Give the numbers for the comparison."
        )
    return _run_tool_directly(
        "perturbation_solve",
        body.model_dump(mode="json"),
        conversation_id,
        user,
        db,
        engine,
        kb,
        limiter,
        settings,
    )


@router.post(
    "/conversations/{conversation_id}/tools/geometry",
    response_model=MessageOut,
    status_code=status.HTTP_201_CREATED,
)
def geometry_directly(
    conversation_id: uuid.UUID,
    body: GeometryArgs,
    user: UserDep,
    db: DbDep,
    engine: EngineDep,
    kb: KbDep,
    limiter: LimiterDep,
    settings: SettingsDep,
) -> MessageOut:
    """Ask for more about the geometry on a card (its request is posted back unchanged)."""
    return _run_tool_directly(
        "geometry",
        body.model_dump(mode="json"),
        conversation_id,
        user,
        db,
        engine,
        kb,
        limiter,
        settings,
    )


class DistributionAnswer(DistributionArgs):
    """The call of a distribution card with the user's answer to "is that a Dirac delta?"."""

    notation: dict[NotationName, Reading] = Field(default_factory=dict)

    @field_validator("notation")
    @classmethod
    def _sensible_readings(cls, value: dict[str, str]) -> dict[str, str]:
        return check_readings(value)


@router.post(
    "/conversations/{conversation_id}/tools/distribution",
    response_model=MessageOut,
    status_code=status.HTTP_201_CREATED,
)
def distribution_directly(
    conversation_id: uuid.UUID,
    body: DistributionAnswer,
    user: UserDep,
    db: DbDep,
    engine: EngineDep,
    kb: KbDep,
    limiter: LimiterDep,
    settings: SettingsDep,
) -> MessageOut:
    """Run a distribution card again with what the user answered (the model is not involved)."""
    return _run_tool_directly(
        "distribution",
        body.model_dump(mode="json", exclude={"notation"}),
        conversation_id,
        user,
        db,
        engine,
        kb,
        limiter,
        settings,
        notation={k: v for k, v in body.notation.items()},
    )


class VariationAnswer(VariationInput):
    """A variation card's call with the numbers the USER entered in its form."""


NUMERIC_FIELDS = (
    "action",
    "values",
    "definitions",
    "paths",
    "sweep",
    "perturbations",
    "automatic_perturbations",
    "with_second",
)


@router.post(
    "/conversations/{conversation_id}/tools/variation",
    response_model=MessageOut,
    status_code=status.HTTP_201_CREATED,
)
def variation_directly(
    conversation_id: uuid.UUID,
    body: VariationAnswer,
    user: UserDep,
    db: DbDep,
    engine: EngineDep,
    kb: KbDep,
    limiter: LimiterDep,
    settings: SettingsDep,
) -> MessageOut:
    """Run a variation card with the user's numbers (the model is not involved and cannot do this)."""
    arguments = body.model_dump(mode="json", include=set(VariationArgs.model_fields) - {"action"})
    numeric = body.model_dump(mode="json", include=set(NUMERIC_FIELDS))
    return _run_tool_directly(
        "variation",
        arguments,
        conversation_id,
        user,
        db,
        engine,
        kb,
        limiter,
        settings,
        numeric=numeric,
    )


class StatMechAnswer(StatMechArgs):
    """A statistical-mechanics card's call with what the USER chose: the phase-space cell, whether the
    copies are indistinguishable, the numbers for an evaluation and a sweep."""

    phase_space_cell: str | None = Field(default=None, max_length=64)
    identical: bool | None = None
    values: dict[str, str] = Field(default_factory=dict, max_length=16)
    sweep: Sweep | None = None


@router.post(
    "/conversations/{conversation_id}/tools/statistical_mechanics",
    response_model=MessageOut,
    status_code=status.HTTP_201_CREATED,
)
def statmech_directly(
    conversation_id: uuid.UUID,
    body: StatMechAnswer,
    user: UserDep,
    db: DbDep,
    engine: EngineDep,
    kb: KbDep,
    limiter: LimiterDep,
    settings: SettingsDep,
) -> MessageOut:
    """Run a statistical-mechanics card again with the user's choices (the model is not involved)."""
    answer_fields = {"phase_space_cell", "identical", "values", "sweep"}
    answers = {
        k: v
        for k, v in body.model_dump(mode="json", include=answer_fields).items()
        if v not in (None, {}, [])
    }
    return _run_tool_directly(
        "statistical_mechanics",
        body.model_dump(mode="json", exclude=answer_fields),
        conversation_id,
        user,
        db,
        engine,
        kb,
        limiter,
        settings,
        answers=answers,
    )


LEGACY_QUOTED = "from an earlier tool result; how it was verified was not recorded"


OLD_RESULT_CHARS = 600  # what is kept of a tool result from an earlier exchange
RECENT_MESSAGES = 4  # the last two exchanges keep their tool results in full


def _shorten(text: str, limit: int = OLD_RESULT_CHARS) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    if "\n" in cut:
        cut = cut[: cut.rindex("\n")]
    return cut + "\n[shortened: an earlier result]"


def _history(
    messages: list[ChatMessage],
    *,
    max_messages: int = HISTORY_MESSAGES,
    max_tokens: int | None = None,
) -> tuple[list[dict[str, Any]], set[str], dict[str, str], dict[str, str], dict[str, list[str]]]:
    """Rebuild the model's view of earlier turns.

    Also returns what earlier turns established, so formulas are marked the same way across turns:
    the formulas verified results produced, the formulas that were only quoted (with why), and the
    verified results that a later call may refer to (`@r1`) with the restrictions each is valid under.
    What was established is read from the WHOLE conversation (so a result id is never reused and its
    restrictions are never lost when the conversation outgrows the window); what the model is shown
    is the newest `max_messages` stored messages, at most `max_tokens` of them, starting at a user
    message, with the tool results of older exchanges shortened. Messages stored before verification
    was recorded per step have only text: their formulas count as quoted, never as verified.
    """
    units: list[list[dict[str, Any]]] = []  # what the model sees of each stored message
    known: set[str] = set()
    quoted: dict[str, str] = {}
    results: dict[str, str] = {}
    restrictions: dict[str, list[str]] = {}
    recent_from = max(0, len(messages) - RECENT_MESSAGES)
    for index, m in enumerate(messages):
        if m.role == "user":
            text = str(m.content.get("text", ""))
            if units and len(units[-1]) == 1 and units[-1][0]["role"] == "user":
                # The reply to the previous message was never stored (the connection dropped): the
                # model sees one message, never two user messages in a row.
                units[-1][0]["content"] += "\n\n" + text
            else:
                units.append([{"role": "user", "content": text}])
            continue
        unit: list[dict[str, Any]] = []
        cards = {
            b["card"]["id"]: b["card"]
            for b in m.content.get("blocks", [])
            if b.get("kind") == "card" and isinstance(b.get("card"), dict)
        }
        for step in m.content.get("trace", []):
            call_id = str(step.get("call_id") or f"call_{step.get('id', uuid.uuid4().hex[:8])}")
            unit.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": call_id, "name": step["name"], "arguments": step["arguments"]}
                    ],
                }
            )
            content = step["for_model"] if index >= recent_from else _shorten(step["for_model"])
            number = cards.get(step["id"], {}).get("number")
            if isinstance(number, int):
                content += (
                    f"\nequation_number=({number}) (the user sees this result as equation ({number}); "
                    f"@e{number} reuses it exactly)"
                )
            result_id = step.get("result_id")
            if result_id:
                content += "\n" + reference_note(result_id)
                latex = cards.get(step["id"], {}).get("data", {}).get("result_latex")
                if isinstance(latex, str) and latex:
                    results[result_id] = latex
                data = cards.get(step["id"], {}).get("data", {})
                carried = [
                    r
                    for key in ("restrictions_latex", "inherited_restrictions_latex")
                    for r in (data.get(key) or [])
                    if isinstance(r, str)
                ]
                if carried:
                    restrictions[result_id] = list(dict.fromkeys(carried))
            unit.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": step["name"],
                    "content": content,
                }
            )
            if "verified_latex" in step or "quoted" in step:
                known.update(normalize(x) for x in step.get("verified_latex", []))
                quoted.update(step.get("quoted", {}))
            else:  # stored before verification was recorded: quoted, never verified
                for seg in split_segments(step["for_model"], set()):
                    if seg.type != "text":
                        quoted.setdefault(normalize(seg.value), LEGACY_QUOTED)
        unit.append({"role": "assistant", "content": replay_prose(m.content)})
        units.append(unit)
    tail = units[-max_messages:]
    if len(tail) < len(units):  # cut by the window: begin at a user message
        while tail and tail[0][0]["role"] != "user":
            tail = tail[1:]
    out = [message for unit in tail for message in unit]
    if max_tokens:
        out, _ = fit_history(out, fixed_chars=0, limit_tokens=max_tokens)
    return out, known, {k: v for k, v in quoted.items() if k not in known}, results, restrictions


def _context_ok(settings: Settings) -> bool:
    """Does the smallest request (system prompt and the core tools) fit the model's window?"""
    from .agent.budget import estimate_tokens, request_chars
    from .agent.loop import _schemas_for
    from .agent.toolsets import prompt_for, tool_names

    toolset = settings.llm_toolset
    prompt = prompt_for([], toolset)
    tools = _schemas_for(tool_names([], toolset))
    reserve = min(settings.llm_num_predict or 4096, settings.llm_num_ctx // 4)
    needed = estimate_tokens(request_chars(prompt, tools, []))
    return bool(needed < settings.llm_num_ctx - reserve)


def _store_stopped(session_factory: Any, conv_id: uuid.UUID, order: list[str | Card]) -> None:
    """Keep what had arrived when the client went away, so the conversation shows it as stopped.

    Without this the user's message would sit with no reply, or a reply they saw would be gone after a
    reload. Nothing arrived: nothing is stored. The partial prose is not checked (the turn ended), so its
    formulas carry no mark but also no claim of verification.
    """
    if not order:
        return
    blocks = [
        Block(kind="card", card=item)
        if isinstance(item, Card)
        else Block(kind="text", segments=split_segments(item, set(), {}))
        for item in order
    ]
    message = AssistantMessage(
        text="\n\n".join(i for i in order if isinstance(i, str)),
        blocks=blocks,
        trace=[],
        truncated=True,
        stop_reason="stopped",
    )
    with session_factory() as session:
        session.add(
            ChatMessage(
                conversation_id=conv_id, role="assistant", content=message.model_dump(mode="json")
            )
        )
        conv = session.get(Conversation, conv_id)
        if conv is not None:
            conv.updated_at = utcnow()
        session.commit()


FAILED_TEXT = (
    "The assistant hit an internal error before it could answer. Your message was kept; send it "
    "again or use Retry."
)


def _store_failed(session_factory: Any, conv_id: uuid.UUID, order: list[str | Card]) -> None:
    """A turn that failed outside the agent loop leaves a message saying so (with what had arrived),
    so a reload and a debug report show the failure instead of a user message with no reply."""
    blocks = [
        Block(kind="card", card=item)
        if isinstance(item, Card)
        else Block(kind="text", segments=split_segments(item, set(), {}))
        for item in order
    ]
    blocks.append(Block(kind="text", segments=split_segments(FAILED_TEXT, set(), {})))
    message = AssistantMessage(
        text="\n\n".join([i for i in order if isinstance(i, str)] + [FAILED_TEXT]),
        blocks=blocks,
        trace=[],
        truncated=True,
        stop_reason="internal_error",
    )
    with session_factory() as session:
        session.add(
            ChatMessage(
                conversation_id=conv_id, role="assistant", content=message.model_dump(mode="json")
            )
        )
        conv = session.get(Conversation, conv_id)
        if conv is not None:
            conv.updated_at = utcnow()
        session.commit()


def _backfill_numbers(db: Any, conversation: Conversation, messages: list[ChatMessage]) -> None:
    """Number the equations of a conversation that predates equation numbers (and was not numbered yet).

    The same reading-order rule as a live reply, so the numbers are the ones it would have had, and they
    are written back so they never change. A conversation with nothing to number is left alone.
    """
    if conversation.equation_counter or not messages:
        return
    table = EquationTable()
    contents = [(m.role, copy.deepcopy(m.content)) for m in messages]
    if not number_stored(contents, table):
        return
    for m, (_, content) in zip(messages, contents, strict=True):
        m.content = content  # a new object: the JSON column notices the change
    conversation.equation_counter = table.counter
    db.commit()


def _conversation_events(previous: list[ChatMessage], new_text: str) -> list[tuple[str, Any]]:
    """The user messages and tool calls of the whole conversation, for the tool groups it needs."""
    events: list[tuple[str, Any]] = []
    for m in previous:
        if m.role == "user":
            events.append(("user", str(m.content.get("text", ""))))
        else:
            for step in m.content.get("trace", []):
                events.append(("tool", (str(step.get("name", "")), step.get("arguments", {}))))
    events.append(("user", new_text))
    return events


MAX_RELATIONS = 60


def _claim_context(
    messages: list[ChatMessage], current: str
) -> tuple[list[str], dict[str, list[str]]]:
    """What earlier turns established that the claim checker may use: formulas written in the
    conversation, in order (the checker keeps only the ones that are definitions, and names them
    whenever a result depends on one), and the assumptions declared in earlier tool calls."""
    texts: list[str] = []
    assumptions: dict[str, list[str]] = {}
    for m in messages[-HISTORY_MESSAGES:]:
        texts.append(str(m.content.get("text", "")))
        if m.role != "user":
            for step in m.content.get("trace", []):
                declared = step.get("arguments", {}).get("assumptions")
                if isinstance(declared, dict):
                    for name, items in declared.items():
                        if isinstance(items, list):
                            assumptions[str(name)] = [str(i) for i in items]
    texts.append(current)
    relations = [
        seg.value
        for text in texts
        for seg in split_segments(text, set())
        if seg.type != "text" and len(seg.value) <= 500
    ]
    return relations[-MAX_RELATIONS:], assumptions


def _declaration_inputs(
    messages: list[ChatMessage], current: str
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """The tool calls of the whole conversation and every message in order, the current one last."""
    steps: list[dict[str, Any]] = []
    entries: list[tuple[str, str]] = []
    for m in messages:
        entries.append((m.role, str(m.content.get("text", ""))))
        if m.role != "user":
            steps.extend(s for s in m.content.get("trace", []) if isinstance(s, dict))
    entries.append(("user", current))
    return steps, entries


def _prior_matrices(messages: list[ChatMessage], current: str = "") -> list[str]:
    """Matrix and vector names this conversation declared, read generously: in tool calls
    (`matrix_calculus`, `noncommutative` assumptions, `vector_calculus` fields) and in words ("let A
    and B be matrices", any formula of a sentence that says "vector"). The whole conversation counts,
    not only the history window: a name is cheap to remember and forgetting one makes a matrix claim
    checkable as if it were about numbers. A name the user later called a scalar is dropped. This only
    leaves claims unchecked; what refuses a tool is `_declared_for_tools`."""
    steps, entries = _declaration_inputs(messages, current)
    found, cleared = scan_declarations(entries, strict=False)
    return sorted(((matrix_names(steps) | vector_names(steps)) - cleared) | set(found))


def _declared_for_tools(messages: list[ChatMessage], current: str = "") -> dict[str, str]:
    """name -> where it came from, for the names that make a scalar tool refuse: tool-call declarations
    and sentences that *declare* a name (strict reading), minus names the user later called scalars."""
    steps, entries = _declaration_inputs(messages, current)
    found, cleared = scan_declarations(entries, strict=True)
    out = {
        name: ("a matrix or vector declared in a tool call earlier in this conversation")
        for name in (matrix_names(steps) | vector_names(steps)) - cleared
    }
    for name, d in found.items():
        who = "your message" if d.role == "user" else "my earlier reply"
        out[name] = f"{who}: \u00ab{d.quote}\u00bb"
    return out


def _title(text: str) -> str:
    """First words of the first message. Formulas keep their `$`, so the list can typeset them."""
    short = re.sub(r"\s+", " ", text).strip()[:60]
    return short[: short.rfind("$")] if short.count("$") % 2 else short


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/conversations/{conversation_id}/messages")
async def send_message(
    conversation_id: uuid.UUID,
    body: SendMessage,
    request: Request,
    user: UserDep,
    db: DbDep,
    engine: EngineDep,
    kb: KbDep,
    limiter: LimiterDep,
    settings: SettingsDep,
) -> StreamingResponse:
    if settings.llm_mode == "off":
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "The chat assistant is not configured."
        )
    if not limiter.allow(f"chat:{user.id}", settings.chat_rate_per_minute):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Too many messages; slow down.")
    conversation = _owned(db, conversation_id, user.id)
    try:
        params = resolve(settings, body.params, user.email)
    except ParamError as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from exc
    llm, provider_label = resolve_llm(request, settings, user, limiter, body.provider)
    if hasattr(
        llm, "with_params"
    ):  # a turn's parameters are its own: never set on the shared client
        llm = llm.with_params(params)
    previous = list(
        db.scalars(
            select(ChatMessage)
            .where(ChatMessage.conversation_id == conversation.id)
            .order_by(ChatMessage.created_at, ChatMessage.id)
        )
    )
    text = body.content
    reused: ChatMessage | None = None  # retry / edit: the user message is kept, not stored again
    if body.mode != "new":
        last_user = next((m for m in reversed(previous) if m.role == "user"), None)
        if last_user is None:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "There is no message to answer again.")
        cut = previous.index(last_user)
        for later in previous[cut + 1 :]:  # the reply being replaced
            db.delete(later)
        previous = previous[:cut]
        reused = last_user
        if body.mode == "retry":
            text = str(last_user.content.get("text", ""))
    _backfill_numbers(db, conversation, previous)
    equations = build_table(previous, conversation.equation_counter)
    # What "equation 3" in this message means (resolved before this message's own formulas are numbered).
    shown_references, reference_note_text = resolve_references(find_references(text), equations)
    user_equations = number_user_text(text, equations)
    history, prior_known, prior_quoted, prior_results, prior_restrictions = _history(
        previous, max_messages=params.history_messages, max_tokens=params.history_tokens
    )
    prior_groups = conversation_groups(_conversation_events(previous, text))
    prior_unverified = unverified_inputs(previous)
    prior_relations, prior_assumptions = _claim_context(previous, text)
    prior_matrices = _prior_matrices(previous, text)
    declared_for_tools = _declared_for_tools(previous, text)
    if history and history[-1]["role"] == "user":  # the last reply was never stored: one message
        history[-1] = {"role": "user", "content": history[-1]["content"] + "\n\n" + text}
    else:
        history.append({"role": "user", "content": text})
    if reference_note_text:  # for the model only: the stored message stays as the user typed it
        history[-1] = {
            "role": "user",
            "content": history[-1]["content"] + "\n\n" + reference_note_text,
        }

    if not previous:
        conversation.title = _title(text) or conversation.title
    user_content: dict[str, Any] = {"text": text}
    if user_equations:
        user_content["equations"] = user_equations
    if shown_references:
        user_content["references"] = shown_references
    if reused is not None:
        reused.content = user_content  # a new object: the JSON column notices the change
    else:
        db.add(ChatMessage(conversation_id=conversation.id, role="user", content=user_content))
    conversation.equation_counter = equations.counter
    conversation.updated_at = utcnow()
    db.commit()
    user_id, conv_id = user.id, conversation.id
    session_factory = request.app.state.session_factory
    ctx = ToolContext(engine=engine, kb=kb)
    ctx.user_messages = [t for role, t in _declaration_inputs(previous, text)[1] if role == "user"]
    limits = AgentLimits(
        max_steps=settings.agent_max_steps,
        max_seconds=settings.agent_max_seconds,
        max_tool_calls=settings.agent_max_tool_calls,
    )
    trace_id = (
        uuid.uuid4().hex
    )  # one per agent request: joins the stored trace, the events and the log
    info = TurnInfo(
        request_id=str(getattr(request.state, "request_id", "")),
        mode=settings.llm_mode,
        api_version=request.app.version,
        trace_id=trace_id,
        sink=_event_sink(settings.agent_events),
    )
    model_label = (
        llm.name
        if provider_label != "openrouter"
        or llm.name in settings.openrouter_model_list
        or llm.name == settings.openrouter_model
        else "custom"
    )

    async def stream() -> AsyncIterator[str]:
        started = time.monotonic()
        outcome = "error"
        counts: dict[str, Any] = {"steps": 0, "tool_calls": 0, "tool_errors": 0}  # numbers only
        stop_reason = "error"
        arrived: list[
            str | Card
        ] = []  # prose and cards sent so far, in order (kept if the client leaves)
        replied = False
        try:
            if user_equations or shown_references:
                # What the user's message was found to contain, before the reply: numbers for its displayed
                # formulas and what "equation 3" was taken to mean. (The stored message has the same.)
                yield _sse(
                    "user",
                    {"equations": user_equations, "references": shown_references},
                )
            async for event in run_agent(
                llm,
                history,
                ctx,
                limits,
                prior_known=prior_known,
                prior_quoted=prior_quoted,
                prior_relations=prior_relations,
                prior_assumptions=prior_assumptions,
                prior_matrices=prior_matrices,
                declared_for_tools=declared_for_tools,
                prior_results=prior_results,
                prior_restrictions=prior_restrictions,
                prior_groups=prior_groups,
                params=params,
                equations=equations,
                prior_unverified=prior_unverified,
                info=info,
            ):
                if event["event"] == "text":
                    if arrived and isinstance(arrived[-1], str):
                        arrived[-1] += event["data"]["delta"]
                    else:
                        arrived.append(event["data"]["delta"])
                elif event["event"] == "card":
                    arrived.append(Card.model_validate(event["data"]))
                if event["event"] == "message":
                    replied = True
                    stop_reason = str(event["data"].get("stop_reason") or "completed")
                    outcome = "error" if stop_reason in ("model_error", "internal_error") else "ok"
                    trace = event["data"]["trace"]
                    calls = event["data"]["debug"]["llm_calls"]
                    first = [
                        c["first_token_ms"] for c in calls if c.get("first_token_ms") is not None
                    ]
                    prompt = [
                        c["prompt_tokens"] for c in calls if c.get("prompt_tokens") is not None
                    ]
                    done_ = [
                        c["completion_tokens"]
                        for c in calls
                        if c.get("completion_tokens") is not None
                    ]
                    cost = [c["cost_usd"] for c in calls if c.get("cost_usd") is not None]
                    counts = {
                        "steps": len(calls),
                        "tool_calls": len(trace),
                        "tool_errors": sum(1 for t in trace if t.get("status") == "error"),
                    }
                    if first:
                        counts["first_token_ms"] = first[0]
                    if prompt:
                        counts["prompt_tokens"] = sum(prompt)
                    if done_:
                        counts["completion_tokens"] = sum(done_)
                    if cost:
                        counts["cost_usd"] = round(sum(cost), 6)
                elif event["event"] == "error":
                    counts["error_kind"] = event["data"].get("debug", {}).get("error_kind")
                if event["event"] == "message":
                    with session_factory() as session:
                        session.add(
                            ChatMessage(
                                conversation_id=conv_id, role="assistant", content=event["data"]
                            )
                        )
                        conv = session.get(Conversation, conv_id)
                        if conv is not None:
                            conv.updated_at = utcnow()
                            conv.equation_counter = max(conv.equation_counter, equations.counter)
                        session.commit()
                yield _sse(event["event"], {**event["data"], "trace_id": trace_id})
            yield _sse("done", {"trace_id": trace_id})
        except (asyncio.CancelledError, GeneratorExit):
            # The browser went away (Stop, a closed tab, a dropped connection) before the reply was stored.
            stop_reason = "stopped"
            if not replied:
                _store_stopped(session_factory, conv_id, arrived)
            raise
        except Exception:
            log.error("chat_turn_failed", trace_id=trace_id)
            if not replied:
                with contextlib.suppress(Exception):  # never let the record hide the error event
                    _store_failed(session_factory, conv_id, arrived)
            yield _sse(
                "error", {"message": "The assistant hit an internal error.", "trace_id": trace_id}
            )
            yield _sse("done", {"trace_id": trace_id})
        finally:
            log.info(
                "chat_turn",
                status=outcome,
                duration_ms=int((time.monotonic() - started) * 1000),
                user_id=str(user_id),
                trace_id=trace_id,
                stop_reason=stop_reason,
                provider=provider_label,
                model=model_label,
                **counts,
            )

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            # no-transform: a proxy (the web app's, a CDN) must not compress the stream, which makes it
            # arrive in one piece at the end instead of as the reply is written.
            "Cache-Control": "no-store, no-transform",
            "X-Accel-Buffering": "no",
            "X-Trace-ID": trace_id,
        },
    )
