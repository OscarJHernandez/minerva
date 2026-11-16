"""Which tools and rules a conversation sends to the model.

Sending all 33 tool schemas and every rule costs about 22 000 tokens per request, more than a model
host's context window may hold, and the model answers worse the more it has to read. So a conversation
starts with a small core and gains a group of tools when it needs it:

- a tool of the group was already used in this conversation, or the model loaded the group (`load_tools`),
- a keyword of the group appears in a message the user wrote, or
- the model calls a tool of the group directly (the group is loaded and the call runs).

The set only ever grows and keeps the order in which groups were added: the start of the prompt then
stays identical between turns, which lets a model host reuse what it already processed. Nothing is
removed or weakened: `toolset="all"` sends everything, as before.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any, Literal

from .prompts import GROUP_KEYWORDS, GROUP_RULES, GROUP_SUMMARY, GROUP_TOOLS, assemble

Toolset = Literal["auto", "core", "all"]

GROUPS: tuple[str, ...] = tuple(GROUP_RULES)
LOAD_TOOLS = "load_tools"

CORE_TOOLS: tuple[str, ...] = (
    "search_handbook",
    "get_entry",
    "simplify",
    "find_identities",
    "isolate_variable",
    "solve_polynomial",
    "differentiate",
    "series",
    "integrate",
    "creative_telescoping",  # what to try when `integrate` cannot do a parameterised integral
    "substitute",
    "rewrite_part",
)

_GROUP_OF: dict[str, str] = {tool: group for group, tools in GROUP_TOOLS.items() for tool in tools}
_PATTERNS: dict[str, re.Pattern[str]] = {
    group: re.compile(pattern) for group, pattern in GROUP_KEYWORDS.items()
}


def group_of(tool: str) -> str | None:
    """The group a tool belongs to; None for a core tool, `load_tools` or an unknown name."""
    return _GROUP_OF.get(tool)


ALIAS_KINDS = ("case", "group_prefix", "functions_prefix")


def resolve_tool_name(requested: str, known: Iterable[str]) -> tuple[str, str | None]:
    """The registered name a model's tool name stands for, and how it was recognised.

    The rules are closed: surrounding space and letter case, `functions.<tool>`, and `<group>.<tool>` or
    `<group>/<tool>` when that group really contains the tool. Anything else is returned unchanged, so a
    genuinely unknown tool stays unknown (and is refused by `execute`). Alias kinds are a fixed vocabulary.
    """
    registry = set(known)
    if requested in registry:
        return requested, None
    name = requested.strip()
    if name in registry:
        return name, "case"
    lowered = name.lower()
    if lowered in registry:
        return lowered, "case"
    for sep in (".", "/"):
        prefix, found, rest = lowered.partition(sep)
        if not found:
            continue
        if prefix == "functions" and rest in registry:
            return rest, "functions_prefix"
        if rest in GROUP_TOOLS.get(prefix, ()) and rest in registry:
            return rest, "group_prefix"
    return requested, None


def groups_in_text(text: str) -> list[str]:
    """Groups whose keywords appear in a user's message, in registry order."""
    lowered = text.lower()
    return [group for group in GROUPS if _PATTERNS[group].search(lowered)]


def requested_groups(arguments: dict[str, Any]) -> list[str]:
    """The valid group names in the arguments of a `load_tools` call, in the order given."""
    raw = arguments.get("groups", arguments.get("group", []))
    items = [raw] if isinstance(raw, str) else raw if isinstance(raw, list) else []
    found: list[str] = []
    for item in items:
        name = str(item).strip().lower()
        if name in GROUP_RULES and name not in found:
            found.append(name)
    return found


def grow(active: Sequence[str], new: Iterable[str]) -> list[str]:
    """`active` plus the groups of `new` that are not in it yet, appended in order."""
    out = list(active)
    for group in new:
        if group in GROUP_RULES and group not in out:
            out.append(group)
    return out


def conversation_groups(events: Iterable[tuple[str, Any]]) -> list[str]:
    """Groups a conversation has needed so far, in the order it needed them.

    `events` are the stored messages in order: ("user", text) and ("tool", (name, arguments)). A
    user message adds the groups of its keywords first; the tools used in the reply add theirs after.
    """
    active: list[str] = []
    for kind, payload in events:
        if kind == "user":
            active = grow(active, groups_in_text(str(payload)))
        elif kind == "tool":
            name, arguments = payload
            if name == LOAD_TOOLS:
                active = grow(
                    active, requested_groups(arguments if isinstance(arguments, dict) else {})
                )
            elif (group := group_of(str(name))) is not None:
                active = grow(active, [group])
    return active


def tool_names(active: Sequence[str], toolset: Toolset = "auto") -> list[str]:
    """Names of the tools to send, in a fixed order: core, `load_tools`, then groups as added."""
    if toolset == "all":
        return [*CORE_TOOLS, *(t for g in GROUPS for t in GROUP_TOOLS[g])]
    names = list(CORE_TOOLS)
    if toolset == "auto":
        names.append(LOAD_TOOLS)
        for group in active:
            names.extend(GROUP_TOOLS[group])
    return names


def prompt_for(active: Sequence[str], toolset: Toolset = "auto") -> str:
    """The system prompt that goes with `tool_names(active, toolset)`."""
    if toolset == "all":
        return assemble(None)
    return assemble(active if toolset == "auto" else [], load_tools=toolset == "auto")


def summary(group: str) -> str:
    return GROUP_SUMMARY[group]
