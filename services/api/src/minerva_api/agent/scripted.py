"""A deterministic stand-in for the language model.

Used for tests, for developing the chat interface without a model host, and for design review.
It follows simple keyword rules and emits real tool calls, so the tools, cards and verification
marks in the interface are the real ones; only the "thinking" is scripted.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Sequence
from typing import Any

from .llm import Chunk, ToolCall

_DOLLAR = re.compile(r"\$([^$]+)\$")


class ScriptedLLM:
    name = "scripted"

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay

    async def _say(self, text: str) -> AsyncIterator[Chunk]:
        for word in re.findall(r"\S+\s*", text):
            if self.delay:
                await asyncio.sleep(self.delay)
            yield Chunk(text=word)

    async def chat(
        self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]
    ) -> AsyncIterator[Chunk]:
        last = messages[-1]
        if last["role"] == "tool":
            async for chunk in self._after_tool(messages):
                yield chunk
            return
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        lower = user.lower()
        latex = (_DOLLAR.findall(user) or [""])[0].strip()

        if "without tools" in lower or "just tell me" in lower:
            async for c in self._say(
                "Here is the value from memory: $\\int_0^\\infty e^{-x^2}\\,dx = \\frac{\\sqrt{\\pi}}{2}$. "
                "I did not run a tool for this."
            ):
                yield c
        elif "identit" in lower or "split" in lower:
            args: dict[str, Any] = {"latex": latex or "\\log(xy)"}
            if "positive" in lower or not latex:
                args["assumptions"] = {"x": ["positive"], "y": ["positive"]}
            async for c in self._say("I will check which handbook identities apply. "):
                yield c
            yield Chunk(tool_calls=[ToolCall("find_identities", args)])
        elif "simplif" in lower or latex:
            async for c in self._say("I will simplify that and verify the result. "):
                yield c
            yield Chunk(
                tool_calls=[ToolCall("simplify", {"latex": latex or "\\sin^2 x + \\cos^2 x"})]
            )
        elif any(
            w in lower for w in ("gaussian", "formula", "integral", "series", "constant", "law")
        ):
            async for c in self._say("Let me search the handbook. "):
                yield c
            yield Chunk(tool_calls=[ToolCall("search_handbook", {"query": user[:100]})])
        elif "solve" in lower:
            async for c in self._say(
                "I cannot solve equations yet; that tool is not available. I can look for related "
                "methods in the handbook. "
            ):
                yield c
            yield Chunk(tool_calls=[ToolCall("search_handbook", {"query": "polynomial roots"})])
        else:
            async for c in self._say(
                "What would you like to do with it: simplify an expression, look for applicable "
                "identities, or find a formula in the handbook? If it has symbols with properties "
                "(positive, real, integer), tell me which."
            ):
                yield c

    async def _after_tool(self, messages: Sequence[dict[str, Any]]) -> AsyncIterator[Chunk]:
        tool = messages[-1].get("tool_name", "")
        content = str(messages[-1]["content"])
        if tool == "simplify":
            result = re.search(r"result=\$([^$]*)\$", content)
            status = re.search(r"status=(\w+)", content)
            if status and status.group(1) == "ok" and result:
                text = (
                    f"**What I understood:** you want a simpler form.\n\n**Result:** $${result.group(1)}$$\n\n"
                    "Minerva verified it numerically and symbolically (details in the card). "
                    "Do you want me to look for the identities behind it?"
                )
            else:
                text = (
                    "Nothing simpler was found, or the result could not be verified, so I am "
                    "not showing a rewrite. If some symbols are known to be positive or real, tell me "
                    "and I will try again with those assumptions."
                )
        elif tool == "find_identities":
            applicable = [ln for ln in content.splitlines() if ": applicable" in ln]
            blocked = [ln for ln in content.splitlines() if "conditions_not_satisfied" in ln]
            if applicable:
                ident = applicable[0].split(" ")[1]
                text = f"Found an applicable identity: `{ident}`. The card shows where it applies and the verified rewrite."
            elif blocked:
                text = (
                    "A matching identity exists, but its conditions are not provable from what you "
                    "declared, so I did not apply it. Are the variables positive?"
                )
            else:
                text = "No handbook identity applies to this expression."
        elif tool in ("search_handbook", "get_entry"):
            text = "These are the closest handbook entries; the cards show each statement and how it was verified."
        else:
            text = "Done; see the card."
        async for c in self._say(text):
            yield c
