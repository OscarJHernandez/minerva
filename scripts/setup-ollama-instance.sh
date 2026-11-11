#!/usr/bin/env bash
# Sets up a dedicated Ollama server for Minerva on a host you can ssh to (no sudo needed).
#   setup-ollama-instance.sh --ssh user@host --models qwen3.5:35b,gemma4:26b [--port 11440] [--yes]
# What it does, all inside the remote user's home:
#   1. links the listed models from the system Ollama's store into ~/.local/share/minerva/ollama-models
#      (symlinks to its blobs plus a copy of each small manifest: no extra disk, nothing is modified there);
#   2. installs ~/.config/systemd/user/minerva-ollama.service (loopback only, own model slot);
#   3. enables and starts it with `systemctl --user`.
# It never touches the system Ollama or other users' services. Undo with:
#   systemctl --user disable --now minerva-ollama; rm -r ~/.local/share/minerva/ollama-models
set -euo pipefail
SSH_TARGET="" MODELS="" PORT=11440 YES=0
SYSTEM_STORE="/usr/share/ollama/.ollama/models"
while [ $# -gt 0 ]; do
  case "$1" in
    --ssh) SSH_TARGET="${2:?}"; shift 2 ;;
    --models) MODELS="${2:?}"; shift 2 ;;
    --port) PORT="${2:?}"; shift 2 ;;
    --system-store) SYSTEM_STORE="${2:?}"; shift 2 ;;
    --yes) YES=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$SSH_TARGET" ] && [ -n "$MODELS" ] || { echo "usage: $0 --ssh user@host --models a:tag,b:tag [--port N] [--yes]" >&2; exit 2; }
[[ "$PORT" =~ ^[0-9]{4,5}$ ]] || { echo "invalid port" >&2; exit 2; }
[[ "$MODELS" =~ ^[A-Za-z0-9._:,/-]+$ ]] || { echo "invalid model list" >&2; exit 2; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT="$(sed "s/__PORT__/${PORT}/" "$HERE/deploy/ollama/minerva-ollama.service.template")"

echo "On $SSH_TARGET (home directory only, no sudo):"
echo "  - link models from $SYSTEM_STORE: $MODELS"
echo "  - install the user service minerva-ollama on 127.0.0.1:$PORT (own model slot, one model loaded at a time)"
echo "  - the system Ollama and other services are not touched"
if [ "$YES" != 1 ]; then
  read -r -p "Proceed? [y/N] " answer
  [ "$answer" = y ] || { echo "Cancelled."; exit 1; }
fi

ssh "$SSH_TARGET" "python3 - '$SYSTEM_STORE' '$MODELS'" <<'PY'
import json, os, shutil, sys
source, models = sys.argv[1], sys.argv[2].split(",")
dest = os.path.expanduser("~/.local/share/minerva/ollama-models")
os.makedirs(os.path.join(dest, "blobs"), exist_ok=True)
for model in models:
    name, _, tag = model.partition(":")
    tag = tag or "latest"
    manifest = os.path.join(source, "manifests", "registry.ollama.ai", "library", name, tag)
    if not os.path.isfile(manifest):
        sys.exit(f"{model}: not found in {source}")
    target = os.path.join(dest, "manifests", "registry.ollama.ai", "library", name)
    os.makedirs(target, exist_ok=True)
    shutil.copyfile(manifest, os.path.join(target, tag))
    data = json.load(open(manifest))
    digests = [data["config"]["digest"], *[layer["digest"] for layer in data["layers"]]]
    for digest in digests:
        blob = digest.replace(":", "-")
        link = os.path.join(dest, "blobs", blob)
        origin = os.path.join(source, "blobs", blob)
        if not os.path.exists(origin):
            sys.exit(f"{model}: blob {blob} missing in {source}")
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(origin, link)
    print(f"linked {model} ({len(digests)} blobs)")
PY
mkdir_cmd='mkdir -p ~/.config/systemd/user ~/.config/minerva'
ssh "$SSH_TARGET" "$mkdir_cmd && cat > ~/.config/systemd/user/minerva-ollama.service" <<<"$UNIT"
ssh "$SSH_TARGET" 'systemctl --user daemon-reload && systemctl --user enable --now minerva-ollama && sleep 3 && systemctl --user is-active minerva-ollama'
echo "Done. Test on the host: curl -s 127.0.0.1:$PORT/api/tags"
echo "From your machine:      ssh -N -L $PORT:127.0.0.1:$PORT $SSH_TARGET   then   MINERVA_LLM_BASE_URL=http://127.0.0.1:$PORT"
