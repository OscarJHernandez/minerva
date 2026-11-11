#!/usr/bin/env bash
# Builds an image with the rootless BuildKit sidecar of the minerva-runners pod (no Docker daemon).
#   build-image.sh <dockerfile> <image:tag> [push|no-push] [--build-arg KEY=VALUE ...]
# Writes the pushed manifest digest to $DIGEST_FILE (default ./image-digest.txt) when pushing.
# REGISTRY_USER / REGISTRY_TOKEN are required to push. CACHE_REF is an optional registry layer cache.
# Pattern borrowed from the other repositories deployed on the same cluster.
set -euo pipefail
DOCKERFILE="${1:?dockerfile path required}"
IMAGE="${2:?image reference required}"
MODE="${3:-no-push}"
shift $(( $# < 3 ? $# : 3 ))
BUILD_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --build-arg) BUILD_ARGS+=(--opt "build-arg:$2"); shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

export PATH="/opt/buildctl:$PATH"
: "${BUILDKIT_HOST:=unix:///run/buildkit/buildkitd.sock}"
export BUILDKIT_HOST

if [ "$MODE" = push ]; then
  : "${REGISTRY_USER:?}" "${REGISTRY_TOKEN:?}"
fi
if [ -n "${REGISTRY_USER:-}" ] && [ -n "${REGISTRY_TOKEN:-}" ]; then
  DOCKER_CONFIG="$(mktemp -d)"
  export DOCKER_CONFIG
  AUTH=$(printf '%s:%s' "$REGISTRY_USER" "$REGISTRY_TOKEN" | base64 | tr -d '\n')
  printf '{"auths":{"ghcr.io":{"auth":"%s"}}}' "$AUTH" > "$DOCKER_CONFIG/config.json"
  trap 'rm -rf "$DOCKER_CONFIG"' EXIT
fi

if [ "$MODE" = push ]; then
  OUTPUT="type=image,name=${IMAGE},push=true,oci-mediatypes=true,unpack=false"
else
  OUTPUT="type=image,name=${IMAGE},push=false"
fi
IMPORT_OPTS=() EXPORT_OPTS=()
if [ -n "${CACHE_REF:-}" ]; then
  IMPORT_OPTS=(--import-cache "type=registry,ref=${CACHE_REF}")
  [ "$MODE" = push ] && EXPORT_OPTS=(--export-cache "type=registry,ref=${CACHE_REF},mode=max,oci-mediatypes=true,image-manifest=true")
fi

META=$(mktemp)
build() {
  buildctl build \
    --frontend dockerfile.v0 \
    --local context=. \
    --local dockerfile="$(dirname "$DOCKERFILE")" \
    --opt filename="$(basename "$DOCKERFILE")" \
    ${IMPORT_OPTS[@]+"${IMPORT_OPTS[@]}"} \
    ${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"} \
    --opt label:org.opencontainers.image.revision="${GITHUB_SHA:-local}" \
    --output "$OUTPUT" \
    --metadata-file "$META" \
    "$@"
}
# Writing the cache must never cost a release: build again without it if the export fails.
if [ ${#EXPORT_OPTS[@]} -gt 0 ]; then
  build "${EXPORT_OPTS[@]}" || { echo "::warning::The layer cache could not be written; building again without it."; build; }
else
  build
fi

if [ "$MODE" = push ]; then
  grep -o '"containerimage.digest": *"[^"]*"' "$META" | sed 's/.*"\(sha256:[^"]*\)"/\1/' > "${DIGEST_FILE:-image-digest.txt}"
  echo "pushed ${IMAGE}@$(cat "${DIGEST_FILE:-image-digest.txt}")"
fi
