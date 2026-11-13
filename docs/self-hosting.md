# Self-hosting Minerva

Minerva is built to run on your own machines. A self-hosted instance keeps your expressions, derivations,
history and conversations in a database you control, and the optional chat assistant can use a language
model you run yourself with [Ollama](https://ollama.com). Nothing here needs an account with any outside
service.

The public demo is a convenience for trying Minerva out. It processes what you type on someone else's
server, so for unpublished or sensitive work run your own instance.

## What runs

| Service | Port (host) | Role |
| --- | --- | --- |
| `web` | 3000 | The app: chat, workbench, handbook, login. Proxies `/api/v1/*` to the API |
| `api` | 8100 | Computation, accounts, history, handbook, chat agent |
| `postgres` | none | Accounts, history and conversations. Not published on the host |
| `site` | 3001 | The static landing page (optional) |

Ports are bound to `127.0.0.1` only. Put a reverse proxy with TLS in front before exposing anything.

## 1. Build and start with Docker Compose

Requirements: Docker with Compose v2 and about 4 GB of free memory.

```bash
git clone https://github.com/OscarJHernandez/minerva.git
cd minerva
cp .env.example .env
```

Edit `.env` and set `POSTGRES_PASSWORD` and `MINERVA_SESSION_SECRET` to random values:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(48))"
```

Build and start:

```bash
docker compose up --build
```

Open <http://localhost:3000> and create an account on the register page. The workbench and the handbook
work now. Chat will say it is not set up until you connect a model (next section).

## 2. Connect your own model (Ollama)

The chat assistant needs a language model behind an Ollama-compatible API. The model plans, asks and
explains; it never computes anything or decides whether a result is correct. Every result in the chat
comes from the same sandboxed engine and handbook the workbench uses, and formulas the model writes that
no tool produced are marked as unchecked.

### a. Install a model with tool calling

Install [Ollama](https://ollama.com/download), then pull a model that supports tool calling. Larger models
call the tools more reliably. [model-evaluation.md](model-evaluation.md) compares the models tried so far
and shows where each one fails; expect to try more than one.

```bash
ollama pull qwen3.5:35b      # or any other tool-capable model you prefer
ollama run qwen3.5:35b "hello"   # check it loads, then exit with /bye
```

Optional but recommended: create a tuned copy called `minerva`. It shares the weights of the base model
and only fixes the generation settings (low temperature, a 16k context window large enough for the tool
descriptions).

```bash
scripts/create-ollama-model.sh qwen3.5:35b
```

### b. Find the address Minerva should use

`MINERVA_LLM_BASE_URL` is the address of Ollama **as seen from the API container**. Inside a container
`localhost` is the container itself, so `http://localhost:11434` does not work with Docker Compose.

| Where Ollama runs | `MINERVA_LLM_BASE_URL` |
| --- | --- |
| Same machine as Docker (macOS, Windows, Linux) | `http://host.docker.internal:11434` |
| Another machine on your network | `http://<that-machine>:11434` |
| API run without Docker, Ollama on the same host | `http://127.0.0.1:11434` |

`host.docker.internal` is built into Docker Desktop, and `docker-compose.yml` adds it on Linux.

On Linux, Ollama listens on `127.0.0.1` by default, which a container cannot reach through
`host.docker.internal`. Make it listen on all interfaces and keep the port off the internet (see
Security below):

```bash
sudo systemctl edit ollama      # add the two lines below, save, then restart
#   [Service]
#   Environment="OLLAMA_HOST=0.0.0.0:11434"
sudo systemctl restart ollama
```

On macOS and Windows, Docker Desktop reaches the host's Ollama as it is. Another machine must run Ollama
with `OLLAMA_HOST` set to an address the Minerva host can reach.

### c. Set three variables and restart the API

In `.env`:

```bash
MINERVA_LLM_MODE=ollama
MINERVA_LLM_BASE_URL=http://host.docker.internal:11434
MINERVA_LLM_MODEL=minerva      # or the base model name, exactly as `ollama list` prints it
```

```bash
docker compose up -d api
```

Check that the API accepted the settings. This endpoint needs no login and does not call the model:

```bash
curl -s http://127.0.0.1:8100/v1/chat/status
# {"enabled":true,"mode":"ollama","model":"minerva","images":false,"providers":{...}}
```

Open <http://localhost:3000> and ask something such as "simplify sin(x)/(1+cos(x)) + (1+cos(x))/sin(x)".
You should see a result card produced by the engine, with its verification, next to the model's
explanation.

Optional settings in `.env`: `MINERVA_LLM_TIMEOUT_SECONDS` (default 120; raise it for large models on
modest hardware), `MINERVA_LLM_THINK=true` (let reasoning models think before replying: sometimes better,
much slower) and `MINERVA_LLM_STRIP_THINKING=true` (hide reasoning some models write into their reply;
delays streaming).

### Troubleshooting

| Symptom | Likely cause and fix |
| --- | --- |
| Chat page says "Chat is not set up on this server" | `MINERVA_LLM_MODE` is `off`. Set it to `ollama` in `.env` and run `docker compose up -d api` (a plain restart does not re-read `.env`) |
| `docker compose logs api` shows "MINERVA_LLM_MODE=ollama needs MINERVA_LLM_BASE_URL and MINERVA_LLM_MODEL" | One of the two is empty. Set both |
| Chat answers "The model host is not reachable." | Wrong address, or Ollama listens only on loopback (Linux). Test from the container: `docker compose exec api python -c "import urllib.request as u; print(u.urlopen('http://host.docker.internal:11434/api/tags', timeout=5).status)"` |
| "The model host answered with status 404" | The model name does not match `ollama list` exactly. Pull or create it, or fix `MINERVA_LLM_MODEL` |
| The model answers in prose, or prints a JSON object such as `{"name": "simplify", ...}` as its reply, and no result card appears | The model is not making real tool calls: it does not support tool calling, or is too small to use it (small general models often do this). Try another model from [model-evaluation.md](model-evaluation.md) |
| Replies time out | Raise `MINERVA_LLM_TIMEOUT_SECONDS`, or use a smaller or faster model |
| Chat works but some formulas are marked "unchecked" | Intended: the model wrote a formula no tool produced. The engine's cards are the results; treat unchecked text as the model's prose |

The chat debug mode (footer checkbox, or `?debug=1`) shows a timeline of what the agent did (what it chose,
whether the arguments were valid, what was verified, why it stopped), every tool call and result, and copies a
report you can attach to an issue. The report contains your conversation: share it only where you would share
the conversation. Every request has a trace id (`X-Trace-ID`); see [observability.md](observability.md) for what the
server log holds (`MINERVA_AGENT_EVENTS`, never content) and how to read a trace.

### Security

- The Ollama API has **no authentication**. Do not publish its port to the internet. Bind it to a private
  network or use a firewall rule that allows only the Minerva host.
- Minerva sends the conversation, the tool descriptions and tool results to the model host you configured,
  and to nothing else. If that host is your own machine, nothing leaves your network.
- Conversations are stored in your PostgreSQL database, private to each account. Server logs carry
  counters, request and trace ids, never expressions, messages or credentials; Minerva itself sends nothing to a
  log or telemetry service.

## 3. Without Docker

Run the API and the web app directly (see the README for `uv` and `pnpm` setup) and export the same
variables for the API process:

```bash
export MINERVA_LLM_MODE=ollama
export MINERVA_LLM_BASE_URL=http://127.0.0.1:11434
export MINERVA_LLM_MODEL=minerva
```

### Optional: OpenRouter (cloud models)

Ollama stays the default. To let users switch to OpenRouter models in the chat with their own key, add
`MINERVA_OPENROUTER_BYOK=true` to `.env` and run `docker compose up -d api`. Keys never leave the user's
browser tab and the API's memory. Full guide, privacy notes and the Kubernetes procedure:
[openrouter-setup.md](openrouter-setup.md).

## 4. Kubernetes

The Helm chart in `deploy/helm/minerva` takes the same settings as `api.llm.*`. See the "Chat model"
section of [../deploy/README.md](../deploy/README.md). To give Minerva its own Ollama server on a shared
GPU machine, see [../deploy/ollama/README.md](../deploy/ollama/README.md).

## 5. Updating and backing up

```bash
git pull
docker compose up --build -d     # the API applies database migrations on start
```

Your data is in the `pgdata` Docker volume. Back it up with
`docker compose exec postgres pg_dump -U minerva minerva > minerva-backup.sql`.
Registration is open by default; the Helm chart can close it (`api.allowRegistration: false`) for a
private instance.
