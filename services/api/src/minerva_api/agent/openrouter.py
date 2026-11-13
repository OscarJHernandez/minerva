"""OpenRouter client (OpenAI-compatible streaming chat completions).

The base URL is a constant: no user input ever chooses where a request goes. The API key lives only in
the `Authorization` header of the request; it is never put into a message, trace, log or error text.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from .llm import Chunk, LLMError, ToolCall, Usage, new_call_id

BASE_URL = "https://openrouter.ai/api/v1"
MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:@+-]{0,99}$")

_MESSAGES = {
    "auth": "OpenRouter rejected the API key. Check that it is correct and has not been revoked.",
    "credits": "OpenRouter says the key has no credits left (or its spending limit was reached).",
    "rate_limit": "OpenRouter is rate limiting this key or model. Wait a moment and try again.",
    "timeout": "OpenRouter did not answer in time.",
    "unavailable": "OpenRouter or the model provider is unavailable right now.",
    "bad_model": "This model was not accepted. Choose another model; it must support tool calling.",
    "bad_reply": "OpenRouter sent a reply Minerva could not read.",
}


def _error(kind: str) -> LLMError:
    return LLMError(_MESSAGES[kind], kind)


def kind_for_status(status: int) -> str:
    if status in (401, 403):
        return "auth"
    if status == 402:
        return "credits"
    if status == 429:
        return "rate_limit"
    if status in (408, 504):
        return "timeout"
    if status in (400, 404, 422):
        return "bad_model"
    return "unavailable"


def openrouter_messages(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Neutral history -> OpenAI shape (ids, `type`, string arguments, `tool_call_id`)."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            out.append(
                {
                    "role": "assistant",
                    "content": m.get("content") or None,
                    "tool_calls": [
                        {
                            "id": c["id"],
                            "type": "function",
                            "function": {
                                "name": c["name"],
                                "arguments": json.dumps(c["arguments"], ensure_ascii=False),
                            },
                        }
                        for c in m["tool_calls"]
                    ],
                }
            )
        elif m.get("role") == "tool":
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m.get("content", "")})
    return out


@dataclass
class _Partial:
    id: str = ""
    name: str = ""
    args: str = ""


def _arguments(raw: str) -> dict[str, Any]:
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _usage(data: dict[str, Any]) -> Usage | None:
    u = data.get("usage")
    if not isinstance(u, dict):
        return None
    cost = u.get("cost")
    return Usage(
        prompt_tokens=u.get("prompt_tokens") if isinstance(u.get("prompt_tokens"), int) else None,
        completion_tokens=(
            u.get("completion_tokens") if isinstance(u.get("completion_tokens"), int) else None
        ),
        cost_usd=float(cost) if isinstance(cost, int | float) else None,
    )


class OpenRouterClient:
    """Streams `/chat/completions` from OpenRouter with the caller's key."""

    provider = "openrouter"

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_seconds: float = 120.0,
        *,
        max_tokens: int = 4096,
        data_collection: str = "deny",
        zdr: bool = False,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not MODEL_ID.match(model):
            raise LLMError(_MESSAGES["bad_model"], "bad_model")
        self._key = api_key
        self.model = model
        self.name = model
        self._timeout = httpx.Timeout(timeout_seconds, connect=10.0)
        self._max_tokens = max_tokens
        self._data_collection = data_collection
        self._zdr = zdr
        self._transport = transport

    def __repr__(self) -> str:  # never show the key, even in a traceback
        return f"OpenRouterClient(model={self.model!r})"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
            "X-Title": "Minerva",
        }

    async def chat(
        self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]
    ) -> AsyncIterator[Chunk]:
        provider_prefs: dict[str, Any] = {"data_collection": self._data_collection}
        if self._zdr:
            provider_prefs["zdr"] = True
        payload = {
            "model": self.model,
            "messages": openrouter_messages(messages),
            "tools": list(tools),
            "stream": True,
            "temperature": 0.2,
            "max_tokens": self._max_tokens,
            "usage": {"include": True},
            "provider": provider_prefs,
        }
        partial: dict[int, _Partial] = {}
        finished = False
        try:
            async with (
                httpx.AsyncClient(
                    timeout=self._timeout, follow_redirects=False, transport=self._transport
                ) as client,
                client.stream(
                    "POST",
                    f"{BASE_URL}/chat/completions",
                    json=payload,
                    headers=self._headers(),
                ) as response,
            ):
                if response.status_code != 200:
                    raise _error(kind_for_status(response.status_code))
                done = False
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or line.startswith(":"):  # blank or keep-alive comment
                        continue
                    if not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if body == "[DONE]":
                        done = True
                        break
                    try:
                        data = json.loads(body)
                    except json.JSONDecodeError as exc:
                        raise _error("bad_reply") from exc
                    if not isinstance(data, dict):
                        raise _error("bad_reply")
                    if "error" in data:
                        err = data["error"] if isinstance(data["error"], dict) else {}
                        code = err.get("code")
                        status = code if isinstance(code, int) else 502
                        raise _error(kind_for_status(status))
                    usage = _usage(data)
                    for choice in data.get("choices") or []:
                        delta = choice.get("delta") or {}
                        text = delta.get("content") or ""
                        for tc in delta.get("tool_calls") or []:
                            slot = partial.setdefault(int(tc.get("index", 0)), _Partial())
                            if tc.get("id"):
                                slot.id = str(tc["id"])
                            fn = tc.get("function") or {}
                            if fn.get("name"):
                                slot.name += str(fn["name"])
                            if fn.get("arguments"):
                                slot.args += str(fn["arguments"])
                        if choice.get("finish_reason"):
                            finished = True
                        if text:
                            yield Chunk(text=text)
                    if usage is not None:
                        yield Chunk(usage=usage)
                if not (done or finished):
                    raise _error("unavailable")
        except httpx.TimeoutException as exc:
            raise _error("timeout") from exc
        except httpx.HTTPError as exc:
            raise _error("unavailable") from exc
        calls = [
            ToolCall(name=p.name, arguments=_arguments(p.args), id=p.id or new_call_id())
            for _, p in sorted(partial.items())
            if p.name
        ]
        if calls:
            yield Chunk(tool_calls=calls)


@dataclass
class ConnectionReport:
    ok: bool
    message: str
    key_label: str = ""
    credit_remaining: float | None = None
    model_found: bool = False
    supports_tools: bool = False


_models_cache: tuple[float, dict[str, bool]] | None = None
MODELS_TTL = 3600.0


def reset_model_cache() -> None:
    global _models_cache
    _models_cache = None


async def _model_tools(client: httpx.AsyncClient, *, use_cache: bool = True) -> dict[str, bool]:
    """Public model list: id -> supports tool calling. Cached for an hour."""
    global _models_cache
    now = time.monotonic()
    if use_cache and _models_cache and now - _models_cache[0] < MODELS_TTL:
        return _models_cache[1]
    response = await client.get(f"{BASE_URL}/models")
    response.raise_for_status()
    table: dict[str, bool] = {}
    for item in response.json().get("data", []):
        table[str(item.get("id"))] = "tools" in (item.get("supported_parameters") or [])
    _models_cache = (now, table)
    return table


async def check_connection(
    api_key: str,
    model: str,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    use_cache: bool = True,
) -> ConnectionReport:
    """Test Connection: validates the key and the model without a paid completion."""
    if not MODEL_ID.match(model):
        return ConnectionReport(False, "That is not a valid model id.")
    timeout = httpx.Timeout(15.0, connect=8.0)
    try:
        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=False, transport=transport
        ) as client:
            key_response = await client.get(
                f"{BASE_URL}/key", headers={"Authorization": f"Bearer {api_key}"}
            )
            if key_response.status_code in (401, 403):
                return ConnectionReport(False, _MESSAGES["auth"])
            if key_response.status_code == 429:
                return ConnectionReport(False, _MESSAGES["rate_limit"])
            if key_response.status_code != 200:
                return ConnectionReport(False, _MESSAGES["unavailable"])
            info = key_response.json().get("data") or {}
            remaining = info.get("limit_remaining")
            credit = float(remaining) if isinstance(remaining, int | float) else None
            table = await _model_tools(client, use_cache=use_cache)
    except (httpx.HTTPError, ValueError):
        return ConnectionReport(False, _MESSAGES["unavailable"])
    if model not in table:
        return ConnectionReport(
            False, "The key works, but OpenRouter does not list this model.", model_found=False
        )
    if not table[model]:
        return ConnectionReport(
            False,
            "The key works, but this model does not advertise tool calling, which Minerva needs.",
            model_found=True,
        )
    if credit is not None and credit <= 0:
        return ConnectionReport(
            False, _MESSAGES["credits"], credit_remaining=credit, model_found=True
        )
    return ConnectionReport(
        True,
        "Connected. The key is valid and the model supports tool calling.",
        credit_remaining=credit,
        model_found=True,
        supports_tools=True,
    )
