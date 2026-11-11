# Contributing to Minerva

Thanks for helping. Priorities, in order: **correctness, simplicity, maintainability, features**.
Small, reviewable pull requests with tests are preferred over large ones.

## Setup

See the README for `uv sync`, `pnpm install` and the commands to run tests. Python 3.13 and Node 20.19+.

## Before you open a pull request

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy packages/engine/src packages/knowledge/src services/api/src
uv run pytest
uv run python -m minerva_knowledge verify
pnpm lint && pnpm typecheck && pnpm test
```

## Mathematical changes

- Every operation must produce real results from a real engine, or return an explicit status
  (`unchanged`, `not_verified`, `unsupported`, ...). Never add a plausible-looking fallback.
- Add regression tests with expected values, negative cases (things that must **not** simplify) and
  assumption cases (the result changes with `positive`, `real`, ...).
- Do not use `eval`, `exec`, `sympify` or `parse_expr` on user input.

## Adding a handbook entry

1. Choose or add a chapter in `packages/knowledge/src/minerva_knowledge/data/chapters.yaml`.
2. Add the entry to a YAML file in `data/entries/`. Required: `id` (unique, dotted), `title`, `kind`,
   `chapter`, `verification`. Then one of:
   - `rule` (pattern variables and `conditions`) for identities the engine may apply;
   - `check` (`derivative`, `antiderivative`, `integral`, `series`, `tensor`, `equation`) with `statement_latex`;
   - `verification: stated` with `statement_latex` for literature records that cannot be checked.
3. State validity conditions explicitly (`conditions`, `validity`). If a rule needs `x > 0`, say so.
4. Run `uv run python -m minerva_knowledge verify`. Entries labelled `symbolic` must have a symbolic zero.
5. **Sources and copyright:** write formulas yourself and cite by locator (`DLMF §4.21`). Do not paste
   text, tables or figures from books or papers unless the licence allows it and you note it in
   `provenance`. Mathematical facts are not copyrightable; a publisher's selection and wording are.
6. Avoid `E`, `I`, `pi`, `oo` as variable names in text syntax (they are constants).

## Dependencies

Prefer small, permissively licensed (MIT/BSD/Apache) libraries. No GPL/AGPL runtime dependencies without
a licence discussion first (see `docs/licensing.md`). Explain why a new dependency is needed.

## Security

Never include secrets, personal data or infrastructure details in issues, commits or logs. Report
vulnerabilities privately (see SECURITY.md).

## Conduct

Be respectful; see CODE_OF_CONDUCT.md.
