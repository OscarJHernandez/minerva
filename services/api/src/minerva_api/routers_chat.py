"""Chat with the agent: conversations, stored messages, and a streamed reply (server-sent events)."""

from __future__ import annotations

import base64
import binascii
import json
import re
import time
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from minerva_engine import MotionRequest
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .agent.events import AssistantMessage, Block, TraceStep
from .agent.llm import LLMClient, LLMError
from .agent.loop import AgentLimits, TurnInfo, run_agent
from .agent.segments import normalize
from .agent.tools import PerturbationArgs, ToolContext, execute
from .agent.vision import FormulaReader, clean_candidates
from .db import ChatMessage, Conversation, utcnow
from .deps import DbDep, EngineDep, KbDep, LimiterDep, SettingsDep, UserDep
from .logging import get_logger

router = APIRouter(prefix="/v1/chat", tags=["chat"])
log = get_logger()

HISTORY_MESSAGES = 20
_MATH = re.compile(r"\$\$(.+?)\$\$|(?<!\\)\$([^$\n]+?)(?<!\\)\$", re.DOTALL)


class StatusOut(BaseModel):
    enabled: bool
    mode: str
    model: str
    images: bool = False  # can formulas be read from a picture ("Add image")


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
def chat_status(request: Request, settings: SettingsDep) -> StatusOut:
    llm = _llm(request)
    return StatusOut(
        enabled=llm is not None,
        mode=settings.llm_mode,
        model=llm.name if llm else "",
        images=llm is not None and _vision(request) is not None,
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
    messages = db.scalars(
        select(ChatMessage)
        .where(ChatMessage.conversation_id == row.id)
        .order_by(ChatMessage.created_at, ChatMessage.id)
    )
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
) -> MessageOut:
    """Run one tool with the user's own numbers (from a card's form), store it as an assistant message.

    The model is not involved, so it never transcribes the numbers. The result is stored holding the card and
    the tool call, so later turns see it exactly as if the model had called the tool. Works with the chat
    model switched off.
    """
    if not limiter.allow(f"compute:{user.id}", settings.compute_rate_per_minute):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Too many requests; slow down.")
    conversation = _owned(db, conversation_id, user.id)
    started = time.monotonic()
    output = execute(name, arguments, ToolContext(engine=engine, kb=kb))
    elapsed = int((time.monotonic() - started) * 1000)
    content = AssistantMessage(
        text="",
        blocks=[Block(kind="card", card=output.card)],
        trace=[
            TraceStep(
                id=output.card.id,
                name=name,
                arguments=arguments,
                for_model=output.for_model,
                duration_ms=elapsed,
                status=output.status,
            )
        ],
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


def _history(messages: list[ChatMessage]) -> tuple[list[dict[str, Any]], set[str]]:
    """Rebuild the model's view of earlier turns, and the formulas Minerva produced in them."""
    out: list[dict[str, Any]] = []
    known: set[str] = set()
    for m in messages[-HISTORY_MESSAGES:]:
        if m.role == "user":
            out.append({"role": "user", "content": str(m.content.get("text", ""))})
            continue
        for step in m.content.get("trace", []):
            out.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": step["name"], "arguments": step["arguments"]}}
                    ],
                }
            )
            out.append({"role": "tool", "tool_name": step["name"], "content": step["for_model"]})
            for found in _MATH.finditer(step["for_model"]):
                known.add(normalize(found.group(1) or found.group(2)))
        out.append({"role": "assistant", "content": str(m.content.get("text", ""))})
    return out, known


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
    llm = _llm(request)
    if llm is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "The chat assistant is not configured."
        )
    if not limiter.allow(f"chat:{user.id}", settings.chat_rate_per_minute):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Too many messages; slow down.")
    conversation = _owned(db, conversation_id, user.id)
    previous = list(
        db.scalars(
            select(ChatMessage)
            .where(ChatMessage.conversation_id == conversation.id)
            .order_by(ChatMessage.created_at, ChatMessage.id)
        )
    )
    history, prior_known = _history(previous)
    history.append({"role": "user", "content": body.content})

    if not previous:
        conversation.title = _title(body.content) or conversation.title
    db.add(
        ChatMessage(conversation_id=conversation.id, role="user", content={"text": body.content})
    )
    conversation.updated_at = utcnow()
    db.commit()
    user_id, conv_id = user.id, conversation.id
    session_factory = request.app.state.session_factory
    ctx = ToolContext(engine=engine, kb=kb)
    limits = AgentLimits(max_steps=settings.agent_max_steps, max_seconds=settings.agent_max_seconds)
    info = TurnInfo(
        request_id=str(getattr(request.state, "request_id", "")),
        mode=settings.llm_mode,
        api_version=request.app.version,
    )

    async def stream() -> AsyncIterator[str]:
        started = time.monotonic()
        outcome = "error"
        counts = {"steps": 0, "tool_calls": 0, "tool_errors": 0}  # numbers only, never content
        try:
            async for event in run_agent(
                llm, history, ctx, limits, prior_known=prior_known, info=info
            ):
                if event["event"] == "message":
                    outcome = "ok"
                    trace = event["data"]["trace"]
                    counts = {
                        "steps": len(event["data"]["debug"]["llm_calls"]),
                        "tool_calls": len(trace),
                        "tool_errors": sum(1 for t in trace if t.get("status") == "error"),
                    }
                    with session_factory() as session:
                        session.add(
                            ChatMessage(
                                conversation_id=conv_id, role="assistant", content=event["data"]
                            )
                        )
                        conv = session.get(Conversation, conv_id)
                        if conv is not None:
                            conv.updated_at = utcnow()
                        session.commit()
                yield _sse(event["event"], event["data"])
            yield _sse("done", {})
        finally:
            log.info(
                "chat_turn",
                status=outcome,
                duration_ms=int((time.monotonic() - started) * 1000),
                user_id=str(user_id),
                **counts,
            )

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )
