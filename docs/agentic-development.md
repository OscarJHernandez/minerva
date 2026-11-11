# Agentic development

Minerva is meant to be developed by people and coding agents together, under the same rules. This page
records what the repository provides for that and where the pattern comes from.

## What is in the repository

| Piece | Where | Purpose |
| --- | --- | --- |
| Rules for every agent | `AGENTS.md` (`CLAUDE.md` imports it with `@AGENTS.md`) | One source of truth for Claude Code, Codex, Cursor and others |
| Skills | `.agents/skills/` (`.claude/skills` symlinks to it) | Task-specific checklists loaded on demand; index in `AGENTS.md` |
| Claude Code settings | `.claude/settings.json` | Format-on-edit hooks, a read-only command allow-list, `git push` denied |
| One entry point | `Makefile` | `make check` equals what CI runs |
| Consistency guard | `scripts/check_agent_files.py` (CI job `python`) | Fails when a skill is unlisted, misnamed or the symlink breaks |
| Workflow safety | `ci.yml` jobs `workflows`, `changes`, `ci-ok` | actionlint and zizmor, path-based job selection, one required check |
| Pinned actions | all workflows | Full commit SHAs with a version comment, kept current by Dependabot |

## Where the pattern comes from

Fifteen open source partners of Stanford CS146S ("The Modern Software Developer", themodernsoftware.dev)
were inspected in November 2026: Browserbase (stagehand), HeyGen (hyperframes), CopilotKit, Semgrep,
OpenHands, Milvus, marimo, Pi, CrewAI, Warp, Vercel (next.js, ai), cmux, Arize Phoenix, Unsloth and
Anyscale (ray). Common practice:

- `AGENTS.md` at the root (15 of 16 repositories); marimo and ray reduce `CLAUDE.md` to an import.
- Skills in `.agents/skills/` exposed to Claude Code (next.js, Phoenix, Warp, ai, OpenHands, cmux).
- `AGENTS.md` lists each skill and when to load it (OpenHands); a README explains when to write one.
- Agent-authored pull requests are disclosed and drafted (marimo, OpenHands, stagehand).
- Format-on-edit hooks and command allow-lists in `.claude/settings.json` (Phoenix, ray).
- Actions pinned to SHAs, `persist-credentials: false`, zizmor or actionlint (Phoenix, CrewAI, CopilotKit).
- Tests that guard the agent files themselves (CrewAI, CopilotKit, cmux).
- Claude in CI for review, triage and audits (Milvus, Phoenix, cmux): deferred here, see the roadmap.

## Adding or changing a skill

Read `.agents/skills/README.md`, add the skill directory, add its row to the `AGENTS.md` table, and run
`uv run python scripts/check_agent_files.py`.
