"""LLM clients. The model only plans, asks and explains; it never produces results."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx


def new_call_id() -> str:
    return "call_" + uuid.uuid4().hex[:16]


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str = field(default_factory=new_call_id)
    # Set when the model host delivered a call that cannot be used as sent: "missing_name",
    # "arguments_not_json" or "arguments_not_object". The arguments are then `{}`, never a guess.
    malformed: str | None = None


@dataclass
class Usage:
    """Token counts, cost and finish reason as the provider reported them (None when it did not).
    Nothing is estimated."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    finish_reason: str | None = None  # stop | length | other


@dataclass
class Chunk:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage | None = None  # on the last chunk of a reply, when the provider reports it


class LLMError(Exception):
    """The model host could not be reached or answered badly (message is safe to show).

    `kind` is one of auth, credits, rate_limit, timeout, unavailable, bad_model, bad_reply, other.
    """

    def __init__(self, message: str, kind: str = "other") -> None:
        super().__init__(message)
        self.kind = kind


class LLMClient(Protocol):
    """Clients receive the neutral history and convert it to their wire format.

    Neutral messages: `{"role": "assistant", "content", "tool_calls": [{"id", "name", "arguments": dict}]}`
    and `{"role": "tool", "tool_call_id", "name", "content"}`.
    """

    name: str
    provider: str

    def chat(
        self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]
    ) -> AsyncIterator[Chunk]: ...


class ThinkFilter:
    """Drops reasoning that a model writes into its reply before a closing `</think>` tag.

    Some reasoning models do this even when thinking is switched off. Text is held back until the tag
    appears, or `limit` characters have passed without one (then it is treated as a normal reply), so
    use it only for models known to leak, because it delays streaming.
    """

    def __init__(self, limit: int = 4000) -> None:
        self._buffer = ""
        self._done = False
        self._limit = limit

    def feed(self, text: str) -> str:
        if self._done:
            return text
        self._buffer += text
        if "</think>" in self._buffer:
            self._done = True
            out = self._buffer.split("</think>", 1)[1].lstrip()
            self._buffer = ""
            return out
        if len(self._buffer) > self._limit:
            self._done = True
            out, self._buffer = self._buffer, ""
            return out
        return ""

    def flush(self) -> str:
        out, self._buffer, self._done = self._buffer, "", True
        return "" if out.lstrip().startswith("<think>") else out


class OllamaClient:
    """Streams `/api/chat` from an Ollama-compatible server (your own model host)."""

    provider = "ollama"

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: float = 120.0,
        think: bool = False,
        strip_thinking: bool = False,
        transport: httpx.AsyncBaseTransport | None = None,
        temperature: float = 0.2,
        seed: int | None = None,
    ) -> None:
        self.strip_thinking = strip_thinking
        # Sampling settings are part of what a benchmark records; the defaults are what chat uses.
        self.temperature = temperature
        self.seed = seed
        self._transport = transport
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = model
        # Reasoning models spend tens of seconds "thinking" before every reply. For a chat that mostly
        # chooses tools, answers are faster without it; set think=True to measure whether it helps.
        self.think = think
        self._timeout = httpx.Timeout(timeout_seconds, connect=10.0)

    async def chat(
        self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]
    ) -> AsyncIterator[Chunk]:
        options: dict[str, Any] = {"temperature": self.temperature}
        if self.seed is not None:
            options["seed"] = self.seed
        payload = {
            "model": self.model,
            "messages": ollama_messages(messages),
            "tools": list(tools),
            "stream": True,
            "think": self.think,
            "options": options,
        }
        try:
            async with (
                httpx.AsyncClient(
                    timeout=self._timeout, follow_redirects=False, transport=self._transport
                ) as client,
                client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response,
            ):
                if response.status_code != 200:
                    raise LLMError(
                        f"The model host answered with status {response.status_code}.",
                        "unavailable",
                    )
                filter_ = ThinkFilter() if self.strip_thinking else None
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise LLMError(
                            "The model host sent an unreadable reply.", "bad_reply"
                        ) from exc
                    message = data.get("message") or {}
                    calls = [_tool_call(c) for c in message.get("tool_calls") or []]
                    text = message.get("content") or ""  # "thinking" is deliberately not forwarded
                    if filter_ is not None:
                        text = filter_.feed(text)
                    usage = _usage(data)
                    if text or calls or usage:
                        yield Chunk(text=text, tool_calls=calls, usage=usage)
                if filter_ is not None and (rest := filter_.flush()):
                    yield Chunk(text=rest)
        except httpx.TimeoutException as exc:
            raise LLMError("The model host timed out.", "timeout") from exc
        except httpx.HTTPError as exc:
            raise LLMError("The model host is not reachable.", "unavailable") from exc


def ollama_messages(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Neutral history -> Ollama's shape (`function` wrapper, `tool_name`, no ids)."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            out.append(
                {
                    "role": "assistant",
                    "content": m.get("content") or "",
                    "tool_calls": [
                        {"function": {"name": c["name"], "arguments": c["arguments"]}}
                        for c in m["tool_calls"]
                    ],
                }
            )
        elif m.get("role") == "tool":
            out.append({"role": "tool", "tool_name": m.get("name", ""), "content": m["content"]})
        else:
            out.append(dict(m))
    return out


def _arguments(raw: Any) -> tuple[dict[str, Any], str | None]:
    """The arguments of a call and, if they could not be used as sent, what was wrong with them."""
    if isinstance(raw, dict):
        return raw, None
    if raw is None:
        return {}, None
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}, "arguments_not_json"
        if isinstance(parsed, dict):
            return parsed, None
        return {}, "arguments_not_object"
    return {}, "arguments_not_object"


def _tool_call(raw: Any) -> ToolCall:
    function = raw.get("function") if isinstance(raw, dict) else None
    if not isinstance(function, dict):
        return ToolCall(name="", arguments={}, malformed="missing_name")
    name = function.get("name")
    if not isinstance(name, str) or not name:
        return ToolCall(name="", arguments={}, malformed="missing_name")
    arguments, defect = _arguments(function.get("arguments"))
    return ToolCall(
        name=name,
        arguments=arguments,
        id=str(raw.get("id") or new_call_id()),
        malformed=defect,
    )


_FINISH = {"stop": "stop", "length": "length", "load": "other", "unload": "other"}


def _usage(data: dict[str, Any]) -> Usage | None:
    """Token counts and finish reason from the last line of an Ollama stream, if it has them."""
    if not data.get("done"):
        return None
    prompt, completion = data.get("prompt_eval_count"), data.get("eval_count")
    reason = data.get("done_reason")
    if prompt is None and completion is None and reason is None:
        return None
    return Usage(
        prompt_tokens=prompt if isinstance(prompt, int) else None,
        completion_tokens=completion if isinstance(completion, int) else None,
        finish_reason=_FINISH.get(str(reason), "other") if reason is not None else None,
    )
