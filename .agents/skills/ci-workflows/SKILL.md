---
name: ci-workflows
description: Edit GitHub Actions workflows for Minerva safely. Use for any change under .github/workflows or .github/actions, runner selection, secrets, permissions, deploy or release steps.
---

# ci-workflows

Trusted code (branches of this repository, `main`) runs on self-hosted runners in the maintainer's
cluster. Fork pull requests must never reach them. Read `docs/security-publication.md` first.

## Rules

1. **Fork safety.** Keep the `runs-on` expression that sends fork PRs to `ubuntu-latest`. Never use
   `pull_request_target` to build, install or run PR code. If a `pull_request_target` job is ever needed
   it may only call the GitHub API and must not check out the PR head.
2. **Least privilege.** Top-level `permissions: contents: read`; grant more per job, only where needed.
3. **Pin actions to a full commit SHA** with a trailing `# vX.Y.Z` comment. Dependabot updates them.
   Use `persist-credentials: false` on checkouts that do not push.
4. **No expression injection.** Pass `${{ github.event.* }}` values (titles, branch names, bodies) through
   `env:`, never inline in `run:`.
5. **Secrets.** Only on `push` to `main` or `workflow_dispatch`, behind the `production` environment. Never
   echo them. No hostnames or cluster details in workflow files; use repository variables.
6. **Concurrency.** PR workflows cancel in progress; deploy never does.
7. **Timeouts.** Every job has `timeout-minutes`.

## Check before committing

```bash
uvx zizmor .github/workflows
actionlint
```

Both must report nothing new. The `lint` job in `ci.yml` runs the same tools. Test shards (`tests`, `tests-api`) must all be listed in `ci-ok.needs`
and keep `fail-fast: false`; a shard count change goes in the matrix and in `MINERVA_TEST_SHARD` together. Each job is
one runner pod, so the jobs of one run plus an overlapping Deploy must fit `maxRunners` in `homelab-kubernetes`.
Use `oss-publication-review` before any change that publishes anything.
