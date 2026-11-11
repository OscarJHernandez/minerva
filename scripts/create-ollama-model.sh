#!/usr/bin/env bash
# Creates the dedicated `minerva` model on an Ollama host from a base model already installed there.
#   create-ollama-model.sh <base-model> [--name minerva] [--ssh user@host] [--yes]
# Without --ssh the local `ollama` command is used. Without --yes it shows what it will do and asks first.
# It only adds a model tag (weights are shared with the base); `ollama rm <name>` removes it again.
set -euo pipefail
BASE="${1:?base model required, e.g. qwen3.5:35b}"; shift
NAME=minerva SSH_TARGET="" YES=0
while [ $# -gt 0 ]; do
  case "$1" in
    --name) NAME="${2:?}"; shift 2 ;;
    --ssh) SSH_TARGET="${2:?}"; shift 2 ;;
    --yes) YES=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$BASE" =~ ^[A-Za-z0-9._:/-]+$ ]] || { echo "invalid model name" >&2; exit 2; }
[[ "$NAME" =~ ^[a-z0-9._-]+$ ]] || { echo "invalid new model name" >&2; exit 2; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELFILE="$(sed "s|__BASE__|${BASE}|" "$HERE/deploy/ollama/Modelfile.template")"
run() { if [ -n "$SSH_TARGET" ]; then ssh "$SSH_TARGET" "$@"; else bash -c "$*"; fi; }

echo "Will create model '$NAME' from '$BASE' ${SSH_TARGET:+on $SSH_TARGET} with:"
echo "$MODELFILE" | grep -v '^#' | grep -v '^$' | sed 's/^/  /'
if [ "$YES" != 1 ]; then
  read -r -p "Proceed? [y/N] " answer
  [ "$answer" = y ] || { echo "Cancelled."; exit 1; }
fi
printf '%s\n' "$MODELFILE" | run "ollama create '$NAME' -f -"
echo "Done. Use MINERVA_LLM_MODEL=$NAME"
