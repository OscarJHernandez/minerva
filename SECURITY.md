# Security policy

## Reporting a vulnerability

Please **do not open a public issue** for security problems. Use GitHub's private vulnerability
reporting ("Security" tab, "Report a vulnerability") on this repository. Include steps to reproduce,
the affected version/commit and the impact you see. We aim to acknowledge reports within a few days.

> Maintainers: enable "Private vulnerability reporting" in the repository settings before the
> repository is made public, and add a monitored contact address here if you want email reports.

## Scope and design notes

- **User expressions are never executed as code.** LaTeX and text are parsed by strict
  parsers into SymPy objects (`packages/engine/src/minerva_engine/latexparse.py`, `textparse.py`).
  `eval`, `exec`, `sympify` and `parse_expr` are not used on user input.
- **Computation is isolated.** Each operation runs in a separate process with a wall-clock timeout,
  CPU/address-space limits (enforced on Linux), a concurrency cap and input size/magnitude checks.
  This is defence in depth, not a security boundary against arbitrary code; deployments should add
  container CPU/memory limits and network policy.
- **Authentication:** Argon2id password hashes; random session tokens stored only as HMAC digests;
  `HttpOnly`, `SameSite=Lax` cookies (`Secure` outside development); same-origin check on
  state-changing requests; per-process rate limiting (use a shared limiter or Cloudflare rules in
  production). The API refuses to start outside development with a placeholder session secret.
- **Privacy:** saved calculations contain user-entered mathematics and are private to the account.
  Logs never contain expressions, emails, passwords or tokens.
- **Supply chain:** dependencies are pinned by lockfile; CI runs `pip-audit`, `pnpm audit` and a secret scan.

## Out of scope

Denial of service by volume (rate limiting and capacity belong to the deployment), social engineering,
and issues in third-party services we do not control.

## Handling secrets

Never commit credentials. `.env` is git-ignored; `.env.example` contains placeholders only. If a secret
is committed by accident, rotate it first, then remove it from history. Enable GitHub secret scanning and
push protection on the repository.
