# Which model answers you

Minerva's language model plans, asks and explains; it never computes. Which model that is changes how well it does
those things (see `docs/reliability/model-comparison.md`), so the interface says so, always, from the model host's own
answer.

## What a user sees

- **The chip above the composer** names the weights, for example `gemma4:26b · local` or `OpenRouter · <model id>`.
  Never the bare alias: the deployed alias (`minerva`) is a name the operator can point at other weights.
- **The line under every reply** names the model that wrote that reply and whether thinking was on for it. A tap shows
  the configured name, base model, family, size, quantisation, digest, the window the host loaded, the reply limit, and
  a sentence saying where the facts come from.
- If the model host did not report what the model is, both say **identity not reported**. Nothing is filled in.
- A provider's model (OpenRouter) is marked as hosted: its weights, quantisation and serving are the provider's and are
  not reported by anyone we control.

## How it is known

- `agent/identity.py` asks the Ollama host (`/api/show`, `/api/tags`, `/api/ps`) for the model: base model, family, size,
  quantisation, digest, whether it can think, the window loaded. Short timeout, never raises.
- `GET /v1/chat/status` carries the default model's identity (cached 30 s: it is for the chip). `GET /v1/chat/model-info`
  (signed in) carries any installed local model's, for the chip when the user picks another.
- **A reply's stamp never uses the cache.** The host is asked at the start of the turn and the digest again when the reply
  is complete (Ollama's chat response carries no digest, so this is the closest the host allows). If the two differ, the
  reply is marked "changed during this reply" with both digests; if the host does not answer, the reply says so. The stamp
  also records whether thinking was requested and the reply limit that was sent.
- The stamp is stored with the reply (`AssistantMessage.model_info`), so an old conversation still says what answered
  it after the alias is re-pointed.

## What it does not do

- It does not say the model is good. Accuracy depends on the model and is measured in `docs/reliability/model-comparison.md`.
- It cannot see inside a hosted provider.
- The digest is shown to the user but is not a log field, and the host's address never reaches a response or a report.
