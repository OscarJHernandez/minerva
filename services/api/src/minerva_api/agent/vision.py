"""Reading formulas from a picture. The vision model only transcribes; it never solves or decides.

What it returns is a proposal: the user confirms it, and the engine's strict parser reports how
Minerva understands each formula (`Engine.read_formulas`). Images are never stored or logged.
"""

from __future__ import annotations

import base64
import json
import re
from typing import Any, Protocol

import httpx

from .llm import LLMError

MAX_FORMULAS = 8

PROMPT = (
    "Transcribe every mathematical formula or equation visible in the image into LaTeX, exactly as "
    "written. Copy what is there: do not solve, simplify, complete, correct or explain anything. "
    "Use one list item per formula or equation, without $ signs or other delimiters. "
    'Answer only with JSON of the form {"formulas": ["..."]}. If the image contains no formula, '
    'answer {"formulas": []}.'
)
_SCHEMA = {
    "type": "object",
    "properties": {"formulas": {"type": "array", "items": {"type": "string"}}},
    "required": ["formulas"],
}


class FormulaReader(Protocol):
    name: str

    async def read(self, image: bytes) -> list[str]: ...


_DELIMITED = (
    re.compile(r"^\$\$(.*)\$\$$", re.DOTALL),
    re.compile(r"^\$(.*)\$$", re.DOTALL),
    re.compile(r"^\\\[(.*)\\\]$", re.DOTALL),
    re.compile(r"^\\\((.*)\\\)$", re.DOTALL),
)


def clean_candidates(items: list[Any], max_chars: int) -> list[str]:
    """Strip math delimiters, drop empty, repeated and over-long entries, keep at most 8."""
    out: list[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        text = item.strip()
        for pattern in _DELIMITED:
            if match := pattern.match(text):
                text = match.group(1).strip()
                break
        if text and len(text) <= max_chars and text not in out:
            out.append(text)
        if len(out) == MAX_FORMULAS:
            break
    return out


class OllamaFormulaReader:
    """One non-streamed `/api/chat` call to a vision-capable model on your own Ollama host."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = model
        self._transport = transport
        self._timeout = httpx.Timeout(timeout_seconds, connect=10.0)

    async def read(self, image: bytes) -> list[str]:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": PROMPT,
                    "images": [base64.b64encode(image).decode("ascii")],
                }
            ],
            "stream": False,
            "think": False,
            "format": _SCHEMA,
            "options": {"temperature": 0},
        }
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, follow_redirects=False, transport=self._transport
            ) as client:
                response = await client.post(f"{self.base_url}/api/chat", json=payload)
        except httpx.HTTPError as exc:
            raise LLMError("The image model host is not reachable.") from exc
        if response.status_code != 200:
            raise LLMError(f"The image model host answered with status {response.status_code}.")
        try:
            content = response.json()["message"]["content"]
            formulas = json.loads(content)["formulas"]
        except (ValueError, KeyError, TypeError) as exc:
            raise LLMError("The image model sent an unreadable reply.") from exc
        if not isinstance(formulas, list):
            raise LLMError("The image model sent an unreadable reply.")
        return formulas  # cleaned by `clean_candidates`


class ScriptedFormulaReader:
    """Development stand-in: ignores the picture and returns one readable and one unreadable formula."""

    name = "scripted"

    async def read(self, image: bytes) -> list[str]:
        return [r"\frac{x^2-1}{x-1}", r"\sin 2x"]
