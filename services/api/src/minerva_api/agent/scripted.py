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
        elif "dimension" in lower and re.search(r"\w+\s*\[[^\]]+\]", user):
            # Development stand-in: quantities are written `name [unit]`, the unknown as `unknown name`.
            quantities = [
                {"name": n, "unit": u.strip()} for n, u in re.findall(r"(\w+)\s*\[([^\]]+)\]", user)
            ]
            found = re.search(r"unknown\s+(\w+)", lower)
            dim_args: dict[str, Any] = {"quantities": quantities}
            if found:
                dim_args["unknown"] = next(
                    (q["name"] for q in quantities if q["name"].lower() == found.group(1)), None
                )
            async for c in self._say("I will find the dimensionless groups and check the units. "):
                yield c
            yield Chunk(tool_calls=[ToolCall("dimensional_analysis", dim_args)])
        elif "units" in lower and latex and re.search(r"\w+\s*\[[^\]]+\]", user):
            quantities = [
                {"name": n, "unit": u.strip()} for n, u in re.findall(r"(\w+)\s*\[([^\]]+)\]", user)
            ]
            async for c in self._say("I will check the dimensions of that equation. "):
                yield c
            yield Chunk(
                tool_calls=[ToolCall("check_units", {"equation": latex, "quantities": quantities})]
            )
        elif "numerically" in lower and latex:
            # Development stand-in: fixed numbers for a unit-mass oscillator, optionally swept in k.
            args: dict[str, Any] = {
                "lagrangian": latex,
                "coordinates": ["x"],
                "parameters": {"m": 1.0, "k": 4.0},
                "initial_positions": [1.0],
                "initial_velocities": [0.0],
                "t_end": 10.0,
            }
            if "sweep" in lower:
                args["parameters"] = {"m": 1.0}
                args["sweep"] = {"name": "k", "values": [1.0, 2.0, 4.0]}
            async for c in self._say(
                "I will integrate the equations of motion and verify the run. "
            ):
                yield c
            yield Chunk(tool_calls=[ToolCall("integrate_motion", args)])
        elif "lagrangian" in lower and latex:
            names = list(dict.fromkeys(re.findall(r"\\dot\s*\{?\\?([A-Za-z]+)", latex)))
            async for c in self._say(
                "I will derive the equations of motion and look for symmetries. "
            ):
                yield c
            yield Chunk(
                tool_calls=[
                    ToolCall(
                        "analyze_lagrangian", {"lagrangian": latex, "coordinates": names or ["x"]}
                    )
                ]
            )
        elif ("differentiat" in lower or "derivative" in lower) and latex:
            order = 2 if "second" in lower else 1
            name = "y" if re.search(r"\bwith respect to y\b|\bin y\b", lower) else "x"
            async for c in self._say("I will differentiate it and check every step. "):
                yield c
            yield Chunk(
                tool_calls=[
                    ToolCall(
                        "differentiate",
                        {"latex": latex, "variables": [{"name": name, "order": order}]},
                    )
                ]
            )
        elif ("integra" in lower or "antiderivative" in lower) and latex:
            name = "y" if re.search(r"\bwith respect to y\b|\bin y\b|\bdy\b", lower) else "x"
            async for c in self._say("I will find an antiderivative and verify it. "):
                yield c
            yield Chunk(tool_calls=[ToolCall("integrate", {"latex": latex, "variable": name})])
        elif "solve" in lower and latex:
            async for c in self._say("I will solve the equation and check the roots. "):
                yield c
            yield Chunk(tool_calls=[ToolCall("solve_polynomial", {"equation": latex})])
        elif "solve" in lower:
            async for c in self._say(
                "Which equation should I solve? Write it with an equals sign, for example "
                "$x^3+x^2+x+1=0$. "
            ):
                yield c
        elif "identit" in lower or "split" in lower:
            args = {"latex": latex or "\\log(xy)"}
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
            found = re.search(r"status=(\w+)", content)
            status = found.group(1) if found else "failed"
            if status == "ok" and result:
                text = (
                    f"**What I understood:** you want a simpler form.\n\n**Result:** $${result.group(1)}$$\n\n"
                    "Minerva verified it numerically and symbolically (details in the card). "
                    "Do you want me to look for the identities behind it?"
                )
            elif status == "unchanged":
                text = (
                    "Nothing simpler was found, so the expression is returned unchanged. If some "
                    "symbols are known to be positive or real, tell me and I will try again."
                )
            else:
                text = f"I could not complete that ({status}). The card says why; tell me how to rephrase it."
        elif tool == "differentiate":
            found = re.search(r"status=(\w+)", content)
            status = found.group(1) if found else "failed"
            if status == "ok":
                text = (
                    "The card shows the derivative and each step. Minerva compared every step with "
                    "an independent numerical derivative at random points."
                )
            else:
                text = (
                    f"I could not differentiate that ({status}). The card says why; for example "
                    "|x| needs x declared real."
                )
        elif tool == "integrate":
            found = re.search(r"status=(\w+)", content)
            status = found.group(1) if found else "failed"
            if status == "ok":
                text = (
                    "The card shows an antiderivative. Minerva differentiated it numerically at "
                    "random points and compared it with your integrand."
                )
            else:
                text = f"I could not integrate that ({status}). The card says why."
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
        elif tool == "analyze_lagrangian":
            text = (
                "The card lists the equations of motion, the accelerations, and the constants of motion "
                "that Minerva verified on those equations, and says which symmetries it did not search for."
            )
        elif tool == "dimensional_analysis":
            text = (
                "The card lists how each unit was read, the dimensionless groups (each checked), "
                "what the dimensions fix about the unknown, and, if its units cannot be built from "
                "the knowns, the dimension that is missing and which constants would supply it."
            )
        elif tool == "check_units":
            text = (
                "The card shows the dimension of each side and, if they differ, the missing factor."
            )
        elif tool == "integrate_motion":
            text = (
                "The card shows the verified trajectory, the drift of the constants of motion and the "
                "phase portrait. Runs that failed verification carry no data."
            )
        elif tool == "solve_polynomial":
            text = (
                "The card shows the factorisation, whether a formula in radicals exists, the verified roots "
                "and the handbook methods that apply."
            )
        elif tool in ("search_handbook", "get_entry"):
            text = "These are the closest handbook entries; the cards show each statement and how it was verified."
        else:
            text = "Done; see the card."
        async for c in self._say(text):
            yield c
