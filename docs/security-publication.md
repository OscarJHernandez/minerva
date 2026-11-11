# Security and publication notes

## Before making the repository public

Run the `oss-publication-review` skill (`.agents/skills/oss-publication-review/SKILL.md`). In short:

1. Audit tracked files and **full git history** for secrets (`gitleaks detect --redact`), `.env` files,
   kubeconfigs, tokens, private hostnames or IPs, personal data, and large generated artifacts.
2. Confirm `.env.example` has placeholders only and `.gitignore` covers local files.
3. Review dependencies and licences (`docs/licensing.md`), confirm no GPL/AGPL runtime dependency.
4. Confirm handbook content contains no copied copyrighted text (locator-only citations).
5. Enable GitHub secret scanning, push protection, Dependabot alerts and private vulnerability reporting.
6. Set branch protection on `main`; require the CI checks.
7. Replace the placeholder contact paragraphs in `SECURITY.md` and `CODE_OF_CONDUCT.md`.
8. Get the maintainer's explicit approval. Nothing is pushed or published by automation.

## Self-hosted CI runners and a public repository

CI is designed to run on self-hosted runners in the homelab cluster. **A public repository plus
self-hosted runners lets anyone who can open a pull request attempt to run code on your hardware.**
Mitigations to apply before going public:

- Settings → Actions → General → "Fork pull request workflows": **require approval for all outside
  collaborators**, and do not send secrets or write tokens to fork workflows.
- Run pull-request jobs only from approved contributors; keep deploy jobs on `push` to `main`, behind a
  protected `production` environment.
- Runners must be ephemeral (the ARC scale sets used here are one pod per job) with no access to cluster
  credentials, a restricted network policy and no mounted host paths.
- Alternative: run fork pull requests on GitHub-hosted runners only and keep self-hosted runners for `main`.
- Workflows grant `contents: read` by default and nothing else unless a job needs it.

## Secret handling

- Real configuration is supplied through environment variables, Kubernetes secrets and repository
  secrets (`KUBE_CONFIG_B64`, tunnel token); none appear in this repository.
- If a secret is ever committed: rotate it first, then rewrite history, then audit access logs.
