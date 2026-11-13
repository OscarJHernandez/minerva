# Using OpenRouter with Minerva

Minerva's chat can run on your own Ollama model (the default, nothing leaves your server) or on a cloud
model through [OpenRouter](https://openrouter.ai). Only the language model changes. Every result still comes
from Minerva's sandboxed engine and handbook, verified the same way, whichever model asks for it.

**Privacy in one paragraph.** With OpenRouter, the conversation's relevant history and Minerva's tool results
are sent to OpenRouter and to the company serving the chosen model, and the usage is billed to the key owner.
Minerva asks OpenRouter not to route to providers that store prompts (`data_collection: deny`, optionally
zero-data-retention endpoints only), which reduces retention but is **not** the same as local processing.
Minerva never falls back from Ollama to OpenRouter (or the other way) by itself.

There are two independent ways to use it:

| | Who pays | Where the key lives | Needs a redeploy |
| --- | --- | --- | --- |
| **A. In the app (BYOK)** | The user | The user's browser tab (memory only) | Only once, to switch the option on |
| **B. Deployment default** | The operator | A Kubernetes Secret | Yes (one workflow run) |

Both can be on at once next to Ollama; users then switch inside the chat.

---

## Workflow A: switch inside the app with your own key

Prerequisite: the operator enabled personal keys (`MINERVA_OPENROUTER_BYOK=true`; for the Helm deployment see
step B-3 and set `byok: true`). Without it, the chat has no OpenRouter option.

1. Create a key at <https://openrouter.ai/keys>. Give it a **credit limit** you are comfortable with.
2. In Minerva's chat, click the model chip at the bottom of the message box (it reads, for example,
   "Local · minerva"). **Model settings** opens.
3. Under **OpenRouter**, paste the key. It is kept in this browser tab's memory only: not in localStorage,
   not in a cookie, not in the conversation history. Reloading the page or logging out forgets it.
4. Pick a model from the list, or choose **Custom model id…** and type any id from
   <https://openrouter.ai/models> (choose one that supports tool calling).
5. Click **Test connection**. Expected: "Connected. The key is valid and the model supports tool calling."
   and, if your key has a limit, the credit left. This does not run a paid completion.
6. Read the notice, tick **I understand…**, and click **Use OpenRouter**. The chip now reads
   "OpenRouter · <model>". If the current conversation already has local turns, Minerva asks once before
   sending them to the cloud.
7. Ask something such as `Simplify $\frac{x^2-1}{x-1}$`. You should get a verified result card and a line
   "via OpenRouter · <model>" under the reply.

**Switch back:** open the chip and click **Use local model**. This is immediate and needs no redeploy. A key
you pasted stays in the tab's memory until you reload or log out, so you can switch again without retyping.

Common messages: "rejected the API key" (wrong or revoked key), "no credits left" (top up or raise the key's
limit), "rate limiting" (wait), "this model was not accepted" (pick a model that supports tools).

---

## Workflow B: make OpenRouter the deployment's provider (Kubernetes)

This is for a Helm/Kubernetes deployment released by `.github/workflows/deploy.yml`. Your local `.env` file
does **not** configure the cluster: the settings below go into the cluster Secret and the
`HELM_EXTRA_VALUES` repository variable. The database, accounts and conversations are not touched by any step.

You can choose how far to go. Most installs want **B-3 with `byok: true`** only (users bring keys, Ollama stays
the default). Do the Secret steps only if you want a server-funded key.

Replace `<owner>/<repo>` with your GitHub repository and use your own app hostname where shown.

### B-1. Create an OpenRouter API key

At <https://openrouter.ai/keys> create a key named e.g. `minerva-server` and set a **credit limit**. Keep the
page open for the next step. Never paste the key into chat, an issue, a values file or a GitHub variable.

### B-2. Store it as a Kubernetes Secret (separate from `minerva-secrets`)

Skip this step if you only want users to bring their own keys.

```bash
read -rs -p "OpenRouter key: " OR_KEY; echo
kubectl -n minerva create secret generic minerva-openrouter --from-literal=api-key="$OR_KEY"
unset OR_KEY
```

`read -s` keeps the key out of your shell history and the screen. Expected output:

```text
secret/minerva-openrouter created
```

Check it exists (key names only, no values):

```bash
kubectl -n minerva get secret minerva-openrouter -o go-template='{{range $k,$v := .data}}{{$k}}{{"\n"}}{{end}}'
```

```text
api-key
```

Rotating later: `kubectl -n minerva delete secret minerva-openrouter`, create it again as above, then
`kubectl -n minerva rollout restart deploy/minerva-api`.

### B-3. Set the provider options in `HELM_EXTRA_VALUES`

`HELM_EXTRA_VALUES` already holds your Ollama settings (and maybe node placement). **Keep what is there** and
add the `openrouter` block under `api.llm`. Read the current value first:

```bash
gh variable get HELM_EXTRA_VALUES --repo <owner>/<repo>
```

Example result, keeping Ollama as the default and letting users bring their own key:

```yaml
api:
  llm:
    mode: ollama                       # unchanged: Ollama stays the default
    baseUrl: "http://<your-ollama-host>:11434"
    model: minerva
    openrouter:
      byok: true                       # users can paste their own key in the chat
```

Add the operator key as well (only the listed accounts may spend it, with a daily cap per user):

```yaml
    openrouter:
      byok: true
      existingSecret: minerva-openrouter
      keyUsers: "you@example.org"      # comma-separated emails; "*" would allow every signed-in user
      dailyTurns: 50
      model: openai/gpt-4.1-mini       # default model for the server key
```

Make OpenRouter the **default** instead of Ollama (users without a key of their own and not in `keyUsers`
then see "Add your own OpenRouter API key"; keep `baseUrl`/`model` set to keep the local option):

```yaml
    mode: openrouter
```

Save the whole document (write it to a file, review it, then):

```bash
gh variable set HELM_EXTRA_VALUES --repo <owner>/<repo> < extra-values.yaml
```

Never put a key in this file. The chart refuses an `apiKey` field, and a literal key would be visible in the
variable. Other options: `models` (list shown in the UI), `maxTokens`, `dataCollection` (`deny` or `allow`),
`zdr: true` (zero-data-retention endpoints only; fewer models), `timeoutSeconds`.

### B-4. Deploy through the existing workflow

The change reaches the cluster through the normal deploy (the code from this feature must be merged to
`main`). Push to `main`, or run it by hand:

```bash
gh workflow run deploy.yml --repo <owner>/<repo>
gh run watch --repo <owner>/<repo>
```

The deploy job now starts with a pre-flight step. Expected log lines:

```text
Chart renders with the supplied values.
Secret 'minerva-openrouter' exists and has key 'api-key'.
```

If the Secret or its key is missing, or the values are invalid, the job stops here with an error before
anything is rolled out. The rest is unchanged: `helm upgrade --rollback-on-failure`, then `helm test`, with an
automatic rollback if the test fails.

### B-5. Verify the real connection and a mathematical tool call

1. Pods are running the new revision:

   ```bash
   kubectl -n minerva get pods -l app.kubernetes.io/component=api
   ```

   ```text
   NAME                           READY   STATUS    RESTARTS   AGE
   minerva-api-7d9c6f5b8c-x2k4p   1/1     Running   0          2m
   ```

2. The API reads the key from the Secret (this prints the reference, not the value):

   ```bash
   kubectl -n minerva get deploy minerva-api -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="MINERVA_OPENROUTER_API_KEY")].valueFrom.secretKeyRef}'
   ```

   ```text
   {"key":"api-key","name":"minerva-openrouter"}
   ```

3. The app reports the providers (public endpoint; no secrets in it):

   ```bash
   curl -s https://<your-app-host>/api/v1/chat/status | python3 -m json.tool | head -30
   ```

   Look for `"openrouter": { "available": true, "byok": true, ... }` and `"default": "ollama"` (or
   `"openrouter"` if you chose that).
4. Open the chat, choose the model chip → OpenRouter, press **Test connection** (expected "Connected."), then
   **Use OpenRouter** and ask `Simplify $\frac{x^2-1}{x-1}$`. You must see a result card with the engine's
   verification and "via OpenRouter · <model>" under the reply.
5. Confirm from the server side (counters only, never content):

   ```bash
   kubectl -n minerva logs deploy/minerva-api | grep chat_turn | tail -1
   ```

   ```text
   {"provider": "openrouter", "model": "openai/gpt-4.1-mini", "status": "ok", "tool_calls": 1, "first_token_ms": 640, "prompt_tokens": 1830, "completion_tokens": 96, "cost_usd": 0.0009, ... }
   ```

### B-6. Switch back to Ollama

- **One user, no redeploy:** chip → **Use local model** (works whenever Ollama is also configured).
- **Deployment default back to Ollama:** set `mode: ollama` (or remove the `mode: openrouter` line) in
  `HELM_EXTRA_VALUES` and run the deploy workflow again.
- **Switch OpenRouter off completely:** remove the `openrouter` block, deploy, and optionally
  `kubectl -n minerva delete secret minerva-openrouter` (and revoke the key at openrouter.ai).
- **Undo the last release:** `helm history minerva -n minerva`, then `helm rollback minerva <revision> -n minerva --wait`.

---

## Self-hosting with Docker Compose

In `.env` (see `.env.example`), keep `MINERVA_LLM_MODE=ollama` with your Ollama settings and add:

```bash
MINERVA_OPENROUTER_BYOK=true
```

then `docker compose up -d api`. For an operator-funded key set `MINERVA_OPENROUTER_API_KEY` and
`MINERVA_OPENROUTER_KEY_USERS=you@example.org`. To use OpenRouter alone: `MINERVA_LLM_MODE=openrouter`,
`MINERVA_OPENROUTER_MODEL=<model id>` and either of the two options above. Chat stays off (workbench and
handbook still work) with `MINERVA_LLM_MODE=off`.

## Comparing providers

```bash
OPENROUTER_API_KEY=... uv run python scripts/eval_models.py --base-url http://<ollama-host>:11434 \
  --models ollama:minerva,openrouter:openai/gpt-4.1-mini --out docs/model-evaluation.md
```

Both run the same cases through the same agent loop. The report adds first-token latency, tool-call and
tool-error counts, results that were not verified, tokens, cost (when OpenRouter reports it) and
rate-limit/timeout counts. OpenRouter runs cost money; pick a small `--cases` list first.

## What is logged

Counters only: provider, model (or `custom`), durations, token counts, cost, tool-call counts, error kind.
Never keys, messages, formulas or tool results. Keys are never written to the database, traces, debug
reports or error messages.
