"""Check that the agent instruction files and skills are consistent.

Rules (see .agents/skills/README.md):
- AGENTS.md exists and CLAUDE.md contains only the line `@AGENTS.md`.
- .claude/skills resolves to .agents/skills.
- Every skill directory has a SKILL.md whose frontmatter `name` equals the directory name and has a
  non-empty `description`, and is linked from the Skills table in AGENTS.md.
- Every skill linked from AGENTS.md exists.

Usage: python scripts/check_agent_files.py [repo_root]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

SKILL_LINK = re.compile(r"\]\(\.agents/skills/([^/)]+)/SKILL\.md\)")


def _frontmatter(text: str) -> dict[str, object] | None:
    match = re.match(r"---\n(.*?)\n---\n", text, re.DOTALL)
    if match is None:
        return None
    try:
        data = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return None
    return data if isinstance(data, dict) else None


def check(root: Path) -> list[str]:
    errors: list[str] = []
    agents = root / "AGENTS.md"
    claude = root / "CLAUDE.md"
    skills_dir = root / ".agents" / "skills"

    if not agents.is_file():
        return ["AGENTS.md is missing"]
    if not claude.is_file() or claude.read_text().strip() != "@AGENTS.md":
        errors.append("CLAUDE.md must contain only the line `@AGENTS.md`")
    link = root / ".claude" / "skills"
    if not link.exists() or link.resolve() != skills_dir.resolve():
        errors.append(".claude/skills must be a symlink to ../.agents/skills")
    if not (skills_dir / "README.md").is_file():
        errors.append(".agents/skills/README.md is missing")

    listed = set(SKILL_LINK.findall(agents.read_text()))
    present: set[str] = set()
    for entry in sorted(skills_dir.iterdir()) if skills_dir.is_dir() else []:
        if not entry.is_dir():
            continue
        present.add(entry.name)
        skill_file = entry / "SKILL.md"
        if not skill_file.is_file():
            errors.append(f"{entry.name}: SKILL.md is missing")
            continue
        meta = _frontmatter(skill_file.read_text())
        if meta is None:
            errors.append(f"{entry.name}: SKILL.md needs valid YAML frontmatter")
            continue
        if meta.get("name") != entry.name:
            errors.append(f"{entry.name}: frontmatter name {meta.get('name')!r} != directory name")
        description = meta.get("description")
        if not isinstance(description, str) or not description.strip():
            errors.append(f"{entry.name}: frontmatter description is missing or empty")
    for name in sorted(present - listed):
        errors.append(f"{name}: not listed in the AGENTS.md Skills table")
    for name in sorted(listed - present):
        errors.append(f"{name}: listed in AGENTS.md but .agents/skills/{name}/ does not exist")
    return errors


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parent.parent
    errors = check(root)
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    if not errors:
        print("agent files: ok")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
