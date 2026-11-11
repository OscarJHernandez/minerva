---
name: oss-publication-review
description: Pre-publication security and licensing audit for the Minerva repository. Use before any public push, making the repository public, tagging a release, or publishing container images or charts. Audits tracked files, git history, configuration, build artifacts, dependencies, licences and CI exposure. Never pushes or publishes by itself.
---

# oss-publication-review

**Hard rule:** this skill only inspects and reports. Do not `git push`, change repository visibility, publish
images or charts, or modify GitHub settings. Report findings and wait for the maintainer's explicit approval.

Everything committed is treated as public. Work through all sections and report PASS / FAIL / NEEDS DECISION
with evidence (file, commit, line), never printing secret values.

## 1. Secrets and sensitive files (tracked files **and** history)

```bash
git ls-files | grep -Ei '(\.env($|\.)|\.pem$|\.key$|kubeconfig|secret|token|credential|\.sqlite|\.db$|\.tfstate|\.tfvars$)' | grep -v '\.env\.example'
git log --all --diff-filter=A --name-only --pretty=format: | sort -u | grep -Ei '(\.env($|\.)|\.pem$|\.key$|kubeconfig|secret|\.sqlite|\.tfstate)' | grep -v '\.env\.example'
gitleaks detect --redact --no-banner            # full history; or: docker run --rm -v "$PWD:/repo" ghcr.io/gitleaks/gitleaks detect -s /repo --redact
grep -rEn '(AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,}|github_pat_|xox[bap]-|-----BEGIN [A-Z ]*PRIVATE KEY-----|eyJhbGciOi)' $(git ls-files)
```

- `.env.example` contains placeholders only; `.env` is untracked (`git check-ignore .env`).
- No kubeconfigs, tunnel tokens, deployer tokens, `KUBE_CONFIG_B64`, registry credentials.
- No private hostnames, IP addresses, node names or cluster details (`git grep -nEi 'alpharelay|192\.168\.|\.local\b|thinkcentre'`).
  The Helm chart and workflows must only reference repository variables for those.
- No user data or computation history (databases, dumps, fixtures built from real accounts).
- Build artifacts and caches are not tracked (`.next`, `out`, `node_modules`, `.venv`, `*.sqlite`, coverage).

If a secret was ever committed: **rotate it first**, then rewrite history, then audit usage.

## 2. Licence and content

- `LICENSE` present; `docs/licensing.md` current. No GPL/AGPL runtime dependency; Cadabra/Sage not imported,
  vendored or bundled in any distributed artifact.
- Re-collect dependency licences (commands in `docs/licensing.md`); investigate anything new that is not
  MIT/BSD/Apache/ISC/PSF/MPL-file-level/OFL.
- Handbook entries: `provenance` is `original` or `public-domain-fact`; no pasted passages or reproduced
  tables; references are locators only. Sample-read the newest entries.
- Images/fonts: only licences allowing redistribution (fonts are OFL-1.1 via fontsource).

## 3. Code and configuration safety

- `grep -rnE '\b(eval|exec|sympify|parse_expr)\(' packages services` has no hit on user-input paths.
- Defaults are safe: the API refuses production start with a placeholder `MINERVA_SESSION_SECRET` or
  `MINERVA_COOKIE_SECURE=false`; logs contain no user input (the log test passes).
- `uv run pytest`, `ruff`, `mypy`, `python -m minerva_knowledge verify`, `pnpm lint typecheck test build` pass.
- Dependency audit: `uvx pip-audit` on the exported lockfile, `pnpm audit --audit-level high`.

## 4. CI exposure (self-hosted runners)

A public repository plus self-hosted runners means strangers can try to run code on your hardware.
Confirm with the maintainer that:

- Settings → Actions → General → Fork pull request workflows: **require approval for all outside collaborators**;
  secrets are not passed to fork workflows.
- Workflows use `pull_request` (not `pull_request_target`) for untrusted code; fork PRs run only on
  GitHub-hosted runners (as in `ci.yml`); deploy jobs run on `push` to `main` behind the `production` environment.
- Runner pods are ephemeral and have no cluster credentials, host mounts or privileged mode beyond BuildKit.
- Workflow `permissions` are `contents: read` unless a job needs more.

## 5. Repository settings checklist (maintainer)

- Branch protection on `main` with required checks; secret scanning and push protection enabled;
  Dependabot alerts enabled; private vulnerability reporting enabled.
- Placeholder contact paragraphs in `SECURITY.md` and `CODE_OF_CONDUCT.md` replaced with real contacts.
- Repository description and topics set; no private issues or discussions referenced.

## 6. Report

End with a table of findings (PASS / FAIL / NEEDS DECISION), the exact commands run, and the list of
maintainer actions required before publication. Do not publish.
