"""How much of a conversation fits in the model's context window.

A model host that is handed more than its window quietly drops the start of the prompt, and the model
then answers as if the conversation had never happened. So the size of a request is estimated before it
is sent and the oldest whole turns of history are left out, visibly, until it fits. The estimate is
deliberately pessimistic (a model's own tokenizer is not available here); the host's real count of
prompt tokens is recorded after the call and compared with the window.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

# Measured on the deployed model: the system prompt is about 4.2 characters per token, JSON tool
# schemas about 6, maths-heavy messages fewer. 3.5 errs on the side of dropping history too early.
CHARS_PER_TOKEN = 3.5
MESSAGE_OVERHEAD_CHARS = 24  # role markers and separators of a chat template


def estimate_tokens(chars: int) -> int:
    return int(chars / CHARS_PER_TOKEN) + 1


def message_chars(message: dict[str, Any]) -> int:
    chars = len(str(message.get("content") or "")) + MESSAGE_OVERHEAD_CHARS
    for call in message.get("tool_calls") or []:
        chars += len(str(call.get("name", ""))) + len(json.dumps(call.get("arguments", {})))
    return chars


def request_chars(
    system: str, tools: Sequence[dict[str, Any]], messages: Sequence[dict[str, Any]]
) -> int:
    return len(system) + len(json.dumps(tools)) + sum(message_chars(m) for m in messages)


def turns(history: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """History split into turns: each starts at a user message. Leading fragments form a turn."""
    out: list[list[dict[str, Any]]] = []
    for message in history:
        if message.get("role") == "user" or not out:
            out.append([])
        out[-1].append(message)
    return out


def fit_history(
    history: Sequence[dict[str, Any]], *, fixed_chars: int, limit_tokens: int
) -> tuple[list[dict[str, Any]], int]:
    """The newest whole turns of `history` that fit, and how many messages were left out.

    `fixed_chars` is everything else in the request (system prompt, tool schemas, the messages of the
    turn in progress). The last turn (the user's new message) is always kept: if it alone does not fit,
    the caller learns it from the estimate and says so; nothing is cut out of a message.
    """
    groups = turns(history)
    if not groups:
        return [], 0
    kept = [groups[-1]]
    used = fixed_chars + sum(message_chars(m) for m in groups[-1])
    budget = limit_tokens * CHARS_PER_TOKEN
    for group in reversed(groups[:-1]):
        size = sum(message_chars(m) for m in group)
        if used + size > budget:
            break
        kept.append(group)
        used += size
    kept.reverse()
    flat = [m for g in kept for m in g]
    return flat, len(history) - len(flat)
