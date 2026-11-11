# Contributor skills

Skills are on-demand instructions that an agent loads when the task matches. They extend `AGENTS.md`
with checklists and procedures that every session does not need. All agents read this directory;
`.claude/skills` is a symlink to it for Claude Code.

## When to write a skill

Write one when the content is a multi-step procedure or checklist, relevant to specific tasks only, and
self-contained. Do **not** write one for a single rule (put it in `AGENTS.md`) or for facts every session
needs (also `AGENTS.md`). Never duplicate a rule in both places: the skill links to the rule.

## Format

```
.agents/skills/<name>/SKILL.md      # required; <name> must equal the frontmatter name
.agents/skills/<name>/<extra>.md    # optional supporting files, linked from SKILL.md
```

```yaml
---
name: <name>
description: What the skill covers and when to use it. Include file paths and trigger phrases; this is how
  an agent decides to load it.
---
```

## Rules

- Every skill must appear in the Skills table in `AGENTS.md` (CI enforces this with
  `scripts/check_agent_files.py`, along with the frontmatter rules above).
- Commands in a skill must be real: run them before committing the skill.
- Skills never carry secrets, hostnames or infrastructure details; the repository is public.
- Skills that inspect or advise must say so and must not push, publish or change settings.
