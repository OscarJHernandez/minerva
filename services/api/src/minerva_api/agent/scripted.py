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


_GEOMETRIES: dict[str, dict[str, Any]] = {
    "polar": {
        "coordinates": ["r", "theta"],
        "metric": [["1", "0"], ["0", "r^2"]],
        "assumptions": {"r": ["positive"]},
    },
    "sphere": {
        "coordinates": ["theta", "phi"],
        "metric": [["a^2", "0"], ["0", "a^2 \\sin^2\\theta"]],
        "assumptions": {"a": ["positive"]},
    },
    "schwarzschild": {
        "coordinates": ["t", "r", "theta", "phi"],
        "metric": [
            ["-(1-2M/r)", "0", "0", "0"],
            ["0", "(1-2M/r)^{-1}", "0", "0"],
            ["0", "0", "r^2", "0"],
            ["0", "0", "0", "r^2 \\sin^2\\theta"],
        ],
        "assumptions": {"M": ["positive"], "r": ["positive"]},
    },
}
_GEOMETRY_WORDS = (
    ("christoffel", "christoffel"),
    ("riemann", "riemann"),
    ("ricci scalar", "ricci_scalar"),
    ("ricci", "ricci"),
    ("einstein", "einstein"),
    ("kretschmann", "kretschmann"),
    ("geodesic", "geodesics"),
)


def _scripted_geometry(lower: str) -> dict[str, Any]:
    name = next((n for n in _GEOMETRIES if n in lower), "polar")
    quantities = [q for word, q in _GEOMETRY_WORDS if word in lower] or ["christoffel"]
    if "ricci scalar" in lower and "ricci" in quantities:
        quantities.remove("ricci")
    return {**_GEOMETRIES[name], "quantities": quantities}


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
        elif any(w in lower for w in ("christoffel", "geodesic", "curvature", "riemann", "ricci")):
            # Development stand-in: three named metrics; a real model asks the user for the metric.
            geometry_args = _scripted_geometry(lower)
            async for c in self._say("I will compute that for the metric you named. "):
                yield c
            yield Chunk(tool_calls=[ToolCall("geometry", geometry_args)])
        elif "matrix" in lower and latex:
            # Development stand-in: every capital letter is a 2x2 abstract matrix; the action comes from
            # the wording. "A and B commute" declares that pair. A real model asks for sizes first.
            bare = re.sub(r"\\[A-Za-z]+", "", latex)
            letters = sorted(set(re.findall(r"[A-Z]", bare)))
            mat_args: dict[str, Any] = {
                "latex": latex,
                "matrices": [{"name": n, "rows": 2} for n in letters],
                "action": "simplify",
            }
            var = next((v for v in "wtxs" if v in bare), "t")
            bounds = re.search(r"from\s+(-?[\w\\]+)\s+to\s+(-?[\w\\]+)", lower)
            if "integra" in lower:
                mat_args.update(action="integrate", variable=var)
                if bounds:
                    mat_args.update(lower=bounds.group(1), upper=bounds.group(2))
            elif "deriv" in lower or "differentiate" in lower:
                mat_args.update(action="differentiate", variable=var)
            if pair := re.search(r"\b([A-Z]) and ([A-Z]) commute", user):
                mat_args["commuting"] = [[pair.group(1), pair.group(2)]]
            async for c in self._say(
                "I will treat these as 2x2 matrices and run the matrix calculus. "
            ):
                yield c
            yield Chunk(tool_calls=[ToolCall("matrix_calculus", mat_args)])
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
        elif ("perturbation" in lower or "lindstedt" in lower or "period" in lower) and latex:
            names = list(dict.fromkeys(re.findall(r"\\dot\s*\{?\\?([A-Za-z]+)", latex)))
            order = int(m.group(1)) if (m := re.search(r"order (\d+)", lower)) else 2
            small = m.group(1) if (m := re.search(r"small parameter (\w+)", user)) else None
            args3: dict[str, Any] = {
                "lagrangian": latex,
                "coordinates": names or ["x"],
                "order": order,
            }
            if small:
                args3["small_parameter"] = small
            if "compare" in lower:
                # Development stand-in: numbers are read from "name=value" pairs in the message, never invented.
                given = {
                    k: float(v)
                    for k, v in re.findall(
                        r"(\w+)\s*=\s*(-?\d+(?:\.\d+)?)", re.sub(r"\$[^$]*\$", "", user)
                    )
                }
                spec = {
                    "epsilon": given.pop("epsilon", 0.1),
                    "amplitude": given.pop("amplitude", None),
                    "t_end": given.pop("t_end", 20.0),
                    "parameters": given,
                }
                args3["compare"] = {k: v for k, v in spec.items() if v is not None}
            async for c in self._say(
                "I will solve it by perturbation theory and verify the result. "
            ):
                yield c
            yield Chunk(tool_calls=[ToolCall("perturbation_solve", args3)])
        elif ("equilibrium" in lower or "small oscillation" in lower) and latex:
            names = list(dict.fromkeys(re.findall(r"\\dot\s*\{?\\?([A-Za-z]+)", latex)))
            order = int(m.group(1)) if (m := re.search(r"order (\d+)", lower)) else 3
            at = re.search(r"at (\w+)\s*=\s*([\w/*]+)", user)
            args2: dict[str, Any] = {
                "lagrangian": latex,
                "coordinates": names or ["x"],
                "order": order,
            }
            if at:
                args2["equilibrium"] = {at.group(1): at.group(2)}
            async for c in self._say(
                "I will expand the equations of motion about the equilibrium. "
            ):
                yield c
            yield Chunk(tool_calls=[ToolCall("expand_motion", args2)])
        elif ("hidden" in lower or "more conserved" in lower or "runge" in lower) and latex:
            names = list(dict.fromkeys(re.findall(r"\\dot\s*\{?\\?([A-Za-z]+)", latex)))
            angle_names = [n for n in names if n in ("theta", "phi", "varphi", "psi")]
            async for c in self._say(
                "I will search a declared family for constants of motion beyond Noether's easy cases. "
            ):
                yield c
            yield Chunk(
                tool_calls=[
                    ToolCall(
                        "search_invariants",
                        {
                            "lagrangian": latex,
                            "coordinates": names or ["x"],
                            "angles": angle_names,
                        },
                    )
                ]
            )
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
        elif ("series" in lower or "taylor" in lower or "expand" in lower) and latex:
            order = int(m.group(1)) if (m := re.search(r"order (\d+)", lower)) else 4
            about = m.group(1) if (m := re.search(r"about (-?\d+(?:/\d+)?)", lower)) else "0"
            async for c in self._say("I will expand it and verify the coefficients. "):
                yield c
            yield Chunk(
                tool_calls=[
                    ToolCall(
                        "series",
                        {"latex": latex, "variable": "x", "point": about, "order": order},
                    )
                ]
            )
        elif ("integra" in lower or "antiderivative" in lower) and latex:
            name = "y" if re.search(r"\bwith respect to y\b|\bin y\b|\bdy\b", lower) else "x"
            int_args: dict[str, Any] = {"latex": latex, "variable": name}
            bounds = re.search(r"from\s+(-?[\w\\]+)\s+to\s+(-?[\w\\]+)", lower)
            if bounds:
                int_args["lower"], int_args["upper"] = bounds.group(1), bounds.group(2)
            async for c in self._say(
                "I will integrate it and verify the value. "
                if bounds
                else "I will find an antiderivative and verify it. "
            ):
                yield c
            yield Chunk(tool_calls=[ToolCall("integrate", int_args)])
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
        elif "simplif" in lower or "factor" in lower or "expand" in lower or latex:
            goal = "factor" if "factor" in lower else "expand" if "expand" in lower else "simplify"
            async for c in self._say(f"I will {goal} that and verify each step. "):
                yield c
            args = {"latex": latex or "\\sin^2 x + \\cos^2 x"}
            if goal != "simplify":
                args["goal"] = goal
            yield Chunk(tool_calls=[ToolCall("simplify", args)])
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
            elif status == "conditions_not_satisfied":
                text = (
                    "An identity matches, but its conditions are not provable from what you declared, "
                    "so nothing was applied. The card names the conditions: do they hold?"
                )
            elif status == "unchanged":
                text = (
                    "Nothing better was found, so the expression is returned unchanged. If some "
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
        elif tool == "perturbation_solve":
            found = re.search(r"status=(\w+)", content)
            status = found.group(1) if found else "failed"
            if status == "ok":
                text = (
                    "The card shows the frequency shift, the period series and the solution. The truncated "
                    "solution was put into the unexpanded equation of motion and its residual falls at the "
                    "claimed order. It is local: it says nothing about large amplitudes. To compare with a "
                    "numerical run I need parameter values, the amplitude and an end time from you."
                )
            else:
                text = (
                    f"I could not solve that perturbatively ({status}). The card says why; this works "
                    "for one coordinate whose acceleration depends on the position only."
                )
        elif tool == "expand_motion":
            found = re.search(r"status=(\w+)", content)
            status = found.group(1) if found else "failed"
            if status == "ok":
                text = (
                    "The card shows the expansion about the equilibrium. Each expansion was checked "
                    "independently of SymPy's series. It is local: it says nothing about large motion."
                )
            else:
                text = (
                    f"I could not expand that ({status}). The card says why; the point must be an "
                    "equilibrium, and L must not depend explicitly on time."
                )
        elif tool == "series":
            found = re.search(r"status=(\w+)", content)
            status = found.group(1) if found else "failed"
            if status == "ok":
                text = (
                    "The card shows the Taylor polynomial. Its coefficients were re-derived with an "
                    "independent contour-integral estimate and the remainder order was confirmed. "
                    "The measured errors are measurements, not guarantees."
                )
            else:
                text = (
                    f"I could not give a series ({status}). The card says why; a pole or branch point "
                    "at the expansion point has no Taylor series."
                )
        elif tool == "integrate":
            found = re.search(r"status=(\w+)", content)
            status = found.group(1) if found else "failed"
            if status == "ok" and "value=$" in content:
                text = (
                    "The card shows the value. Minerva compared it with independent numerical "
                    "quadrature at sampled parameter values."
                )
            elif status == "ok":
                text = (
                    "The card shows an antiderivative. Minerva differentiated it numerically at "
                    "random points and compared it with your integrand."
                )
            elif status == "conditions_not_satisfied":
                text = (
                    "The integral exists only under the conditions on the card, and nothing you "
                    "declared implies them. Do they hold?"
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
        elif tool == "search_invariants":
            text = (
                "The card lists the constants of motion that were verified on the equations of motion, "
                "the exact family that was searched, and what was already known. Nothing outside that "
                "family was searched."
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
        elif tool == "matrix_calculus":
            found = re.search(r"status=(\w+)", content)
            status = found.group(1) if found else "failed"
            result = re.search(r"result=\$([^$]*)\$", content)
            if status == "ok" and "evaluated=NO" in content:
                text = (
                    "Minerva verified an exact reduction, but the integral was **not evaluated**: the card "
                    "shows the equation it satisfies and a block-matrix representation, not entries."
                )
            elif status == "ok" and result:
                text = (
                    f"**Result:** $${result.group(1)}$$\n\n"
                    "The card shows the method, the assumptions used and how it was verified."
                )
            elif status == "conditions_not_satisfied":
                text = (
                    "A step needs a condition that is not established, so nothing was changed. "
                    "The card names it (for example that two matrices commute): does it hold?"
                )
            else:
                text = f"I could not complete that ({status}). The card says why; tell me how to rephrase it."
        elif tool == "geometry":
            found = re.search(r"status=(\w+)", content)
            status = found.group(1) if found else "failed"
            if status == "ok":
                text = (
                    "The card lists each quantity with the conventions used, where the coordinates break "
                    "down, and the checks it passed. Numerical agreement is evidence; a quantity is only "
                    "marked as proved when a symbolic test also succeeded."
                )
            else:
                text = f"I could not complete that ({status}). The card says why; tell me how to rephrase it."
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
