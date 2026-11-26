# syntax=docker/dockerfile:1
# CI runner image for the self-hosted `minerva-runners` scale set: the stock GitHub Actions runner plus CPython 3.13 and
# uv's wheel cache for exactly this repository's uv.lock. Each runner pod starts cold and all of them share one slow
# uplink; with these baked in, `uv sync --frozen` in ci.yml downloads nothing when uv.lock matches the image (and only
# the changed packages when it does not). PR #219 measured why the GitHub Actions cache cannot do this here.
#
#   Build context: repository root.
#   in CI:   .github/workflows/runner-image.yml (pull requests build it, main, a manual run and a monthly schedule push it)
#   use:     arc/values/minerva-runners.yaml in the homelab-kubernetes repository (`image:`, by tag and digest)
#
# Nothing here is required: ci.yml runs unchanged on the stock runner image, only slower. The paths are the ones the
# jobs use: setup-uv sets UV_PYTHON_INSTALL_DIR to ~/.local/share/uv/python, and with its cache off (`auto` on
# self-hosted runners) uv keeps its default cache, ~/.cache/uv. The image holds only open-source packages from uv.lock.
ARG RUNNER_VERSION=2.338.2
FROM ghcr.io/astral-sh/uv:0.13.10 AS uv

FROM ghcr.io/actions/actions-runner:${RUNNER_VERSION}
ARG RUNNER_VERSION
COPY --from=uv /uv /uvx /usr/local/bin/
USER runner
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
ENV UV_PYTHON_INSTALL_DIR=/home/runner/.local/share/uv/python

# The two syncs ci.yml runs (dev environment; production API environment), into throwaway environments, from a copy of
# the manifests and workspace sources: only the cache and the interpreter stay in the image. Hatchling (the build
# backend of the workspace packages) is cached on the way. The second pass is the smoke test: offline, it must need
# nothing that is not already here, so an image that would still download fails this build and not a CI run.
RUN --mount=type=bind,target=/mnt/repo \
    uv python install 3.13 \
 && mkdir /tmp/repo \
 && cp -r /mnt/repo/pyproject.toml /mnt/repo/uv.lock /mnt/repo/.python-version /mnt/repo/packages /mnt/repo/services /tmp/repo/ \
 && cd /tmp/repo \
 && for offline in "" --offline; do \
      UV_PROJECT_ENVIRONMENT=/tmp/dev-venv uv sync --frozen $offline; \
      UV_PROJECT_ENVIRONMENT=/tmp/prod-venv uv sync --frozen --no-dev --no-editable --package minerva-api $offline; \
      rm -rf /tmp/dev-venv /tmp/prod-venv; \
    done \
 && printf 'runner=%s\nuv=%s\nuv_lock_sha256=%s\n' "${RUNNER_VERSION}" "$(uv --version | cut -d' ' -f2)" \
      "$(sha256sum uv.lock | cut -d' ' -f1)" > /home/runner/.minerva-runner-image \
 && cd / && rm -rf /tmp/repo \
 && du -sh /home/runner/.cache/uv /home/runner/.local/share/uv/python

LABEL org.opencontainers.image.source="https://github.com/OscarJHernandez/minerva" \
      org.opencontainers.image.description="Minerva CI runner: GitHub Actions runner with CPython 3.13 and the uv cache for uv.lock baked in" \
      org.opencontainers.image.licenses="Apache-2.0"
