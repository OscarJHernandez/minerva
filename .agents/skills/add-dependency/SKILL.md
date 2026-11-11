---
name: add-dependency
description: Add or upgrade a Python or JavaScript dependency in Minerva with the licence, lockfile and audit checks. Use for any change to pyproject.toml, package.json, uv.lock or pnpm-lock.yaml.
---

# add-dependency

Prefer the standard library or an existing dependency. Every new dependency is public supply chain.

## 1. Justify

Write down: what it is needed for, what already in the repo was considered, maintenance status (recent
releases, maintainers), and its licence.

## 2. Licence gate

- Allowed: MIT, BSD, Apache-2.0, ISC, PSF, MPL-2.0 (file level), OFL for fonts.
- **No GPL or AGPL runtime dependency.** Cadabra 2 and SageMath are not dependencies (`docs/licensing.md`).
- LGPL only as an unmodified installed library, and record it in `docs/licensing.md`.
- Anything unclear: stop and ask the maintainer.

## 3. Add it

```bash
uv add --package <member> <dep>             # member: minerva-engine | minerva-knowledge | minerva-api
pnpm --filter <pkg> add <dep>               # pkg: @minerva/web | @minerva/site | @minerva/ui
```

Commit the lockfile (`uv.lock`, `pnpm-lock.yaml`). Never use `--no-frozen` workarounds in CI.
Pin exact versions for JavaScript packages that render maths or run in the build (as `katex` and `next` are).

## 4. Audit and document

```bash
uv export --frozen --no-hashes --no-emit-workspace > /tmp/req.txt && uvx pip-audit -r /tmp/req.txt --no-deps --disable-pip
pnpm audit --audit-level high
```

Re-run the licence collection from `docs/licensing.md` and update its tables if the licence set changed.
Run `verify-change` afterwards.
