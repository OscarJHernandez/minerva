# A dedicated Ollama server for Minerva

Minerva's chat needs a language model behind an Ollama-compatible API. On a shared machine, loading Minerva's
model must not evict anyone else's, so Minerva gets **its own Ollama server** with its own model slot.

```bash
# 1. Link the models you want to try into a Minerva-only store and start a user service (no sudo):
scripts/setup-ollama-instance.sh --ssh user@host --models qwen3.5:35b,gemma4:26b --port 11440

# 2. Reach it from your machine (it listens on the host's loopback only):
ssh -N -f -L 11440:127.0.0.1:11440 user@host
export MINERVA_LLM_MODE=ollama MINERVA_LLM_BASE_URL=http://127.0.0.1:11440 MINERVA_LLM_MODEL=qwen3.5:35b

# 3. Compare models through the real agent loop:
uv run python scripts/eval_models.py --base-url http://127.0.0.1:11440 --models qwen3.5:35b,gemma4:26b

# 4. Once a base model is chosen, create the `minerva` model (fixed generation settings, shared weights):
OLLAMA_HOST=127.0.0.1:11440 ollama create minerva -f <(sed 's|__BASE__|qwen3.5:35b|' deploy/ollama/Modelfile.template)
```

How it is isolated:

- A **user-level systemd service** (`minerva-ollama`), loopback only, one model loaded at a time, kept for an hour.
- Its model store contains **symlinks** to the system Ollama's blobs plus copies of the small manifests: no extra disk
  and the system store is never modified. Models are added by re-running the setup script.
- The system Ollama and other users' services are untouched.

Remove it: `systemctl --user disable --now minerva-ollama && rm -r ~/.local/share/minerva/ollama-models`.

Notes:

- A user service stops when the user has no session unless lingering is enabled (`loginctl enable-linger <user>`, which
  may need an administrator).
- Loading a model reserves its size in the machine's (shared) GPU memory for as long as it stays loaded.
- To let a cluster reach it, set `OLLAMA_HOST` in `~/.config/minerva/ollama.env`; the endpoint has no authentication,
  so expose it only on a trusted network.
