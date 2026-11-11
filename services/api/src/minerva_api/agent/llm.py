"""LLM clients. The model only plans, asks and explains; it never produces results."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class Chunk:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMError(Exception):
    """The model host could not be reached or answered badly (message is safe to show)."""


class LLMClient(Protocol):
    name: str

    def chat(
        self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]
    ) -> AsyncIterator[Chunk]: ...


class OllamaClient:
    """Streams `/api/chat` from an Ollama-compatible server (your own model host)."""

    def __init__(self, base_url: str, model: str, timeout_seconds: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = model
        self._timeout = httpx.Timeout(timeout_seconds, connect=10.0)

    async def chat(
        self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]
    ) -> AsyncIterator[Chunk]:
        payload = {
            "model": self.model,
            "messages": list(messages),
            "tools": list(tools),
            "stream": True,
            "options": {"temperature": 0.2},
        }
        try:
            async with (
                httpx.AsyncClient(timeout=self._timeout, follow_redirects=False) as client,
                client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response,
            ):
                if response.status_code != 200:
                    raise LLMError(f"The model host answered with status {response.status_code}.")
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise LLMError("The model host sent an unreadable reply.") from exc
                    message = data.get("message") or {}
                    calls = [
                        ToolCall(
                            name=str(c["function"]["name"]),
                            arguments=_arguments(c["function"].get("arguments")),
                        )
                        for c in message.get("tool_calls") or []
                    ]
                    text = message.get("content") or ""  # "thinking" is deliberately not forwarded
                    if text or calls:
                        yield Chunk(text=text, tool_calls=calls)
        except httpx.HTTPError as exc:
            raise LLMError("The model host is not reachable.") from exc


def _arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}
