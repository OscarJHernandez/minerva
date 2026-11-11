"""Chat with the agent: conversations, stored messages, and a streamed reply (server-sent events)."""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .agent.llm import LLMClient
from .agent.loop import AgentLimits, run_agent
from .agent.segments import normalize
from .agent.tools import ToolContext
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
    return StatusOut(enabled=llm is not None, mode=settings.llm_mode, model=llm.name if llm else "")


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
        conversation.title = (
            re.sub(r"\s+", " ", body.content.replace("$", "")).strip()[:60] or conversation.title
        )
    db.add(
        ChatMessage(conversation_id=conversation.id, role="user", content={"text": body.content})
    )
    conversation.updated_at = utcnow()
    db.commit()
    user_id, conv_id = user.id, conversation.id
    session_factory = request.app.state.session_factory
    ctx = ToolContext(engine=engine, kb=kb)
    limits = AgentLimits(max_steps=settings.agent_max_steps, max_seconds=settings.agent_max_seconds)

    async def stream() -> AsyncIterator[str]:
        started = time.monotonic()
        outcome = "error"
        try:
            async for event in run_agent(llm, history, ctx, limits, prior_known=prior_known):
                if event["event"] == "message":
                    outcome = "ok"
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
            )

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )
