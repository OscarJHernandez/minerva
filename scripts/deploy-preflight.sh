#!/usr/bin/env bash
# Pre-flight for the deploy workflow: render the chart exactly as the upgrade will (so a bad value fails
# here, before anything rolls out) and, if the API is configured to read the OpenRouter key from a
# Kubernetes Secret, check that the Secret and key exist. Prints names only, never secret values.
#
#   scripts/deploy-preflight.sh <namespace> <helm args...>      (the same -f / --set arguments as `helm upgrade`)
set -euo pipefail
ns="$1"; shift
rendered="$(helm template minerva deploy/helm/minerva "$@")"
echo "Chart renders with the supplied values."

# `secretKeyRef: { name: "<secret>", key: "<key>" }` on MINERVA_OPENROUTER_API_KEY
ref="$(printf '%s\n' "$rendered" | grep -A1 'name: MINERVA_OPENROUTER_API_KEY' | grep -o 'name: "[^"]*", key: "[^"]*"' || true)"
if [ -z "$ref" ]; then
  echo "No OpenRouter Secret is referenced; nothing more to check."
  exit 0
fi
secret="$(printf '%s' "$ref" | sed -E 's/name: "([^"]*)".*/\1/')"
key="$(printf '%s' "$ref" | sed -E 's/.*key: "([^"]*)".*/\1/')"
if ! command -v kubectl >/dev/null 2>&1; then
  echo "::warning::kubectl not found on the runner; cannot check that Secret '$secret' exists."
  exit 0
fi
if ! keys="$(kubectl -n "$ns" get secret "$secret" -o go-template='{{range $k,$v := .data}}{{$k}}{{"\n"}}{{end}}' 2>/dev/null)"; then
  echo "::error::Secret '$secret' (namespace $ns) does not exist or cannot be read. Create it first: docs/openrouter-setup.md"
  exit 1
fi
if ! printf '%s\n' "$keys" | grep -qx "$key"; then
  echo "::error::Secret '$secret' has no key '$key'. Found keys: $(printf '%s' "$keys" | tr '\n' ' ')"
  exit 1
fi
echo "Secret '$secret' exists and has key '$key'."
