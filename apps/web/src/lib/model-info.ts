import type { ModelInfo } from "./chat-types";

/** The short name a user reads: the weights when the host reported them, never the bare alias alone. */
export function modelLabel(info: ModelInfo | null | undefined, alias = ""): string {
  if (!info) return alias ? `${alias} · identity not reported` : "model not reported";
  if (info.hosted) return `OpenRouter · ${info.alias}`;
  if (info.provider === "scripted") return "scripted stand-in (development)";
  if (!info.known) return `${info.alias} · identity not reported`;
  const base = info.base_model || info.family || info.alias;
  return `${base} · local`;
}

/** Facts for the details view; a fact the host did not report is left out, never filled in. */
export function modelFacts(info: ModelInfo): [string, string][] {
  const thinking =
    info.thinking_requested == null
      ? info.can_think == null
        ? null
        : info.can_think
          ? "supported"
          : "not supported"
      : info.thinking_requested
        ? "on"
        : "off";
  const rows: [string, string | null | undefined][] = [
    [
      "Provider",
      info.hosted ? "OpenRouter (hosted)" : info.provider === "ollama" ? "Local (Ollama)" : info.provider,
    ],
    ["Configured name", info.alias],
    ["Base model", info.base_model],
    ["Family", info.family],
    ["Size", info.parameter_size],
    ["Quantisation", info.quantization],
    ["Digest", info.digest],
    ["Window loaded", info.loaded_context ? `${info.loaded_context} tokens` : null],
    ["Thinking", thinking],
    ["Reply limit", info.reply_budget ? `${info.reply_budget} tokens` : null],
  ];
  return rows.filter((r): r is [string, string] => Boolean(r[1]));
}

export function modelNotes(info: ModelInfo): string[] {
  const notes = [...(info.notes ?? [])];
  if (info.changed_during_reply) {
    notes.push(
      `The model host changed the model while this reply was written (digest ${info.digest ?? "?"} to ${info.digest_after ?? "?"}).`,
    );
  } else if (info.known) {
    notes.push("Weights as reported by the model host when the reply was made.");
  } else if (!info.hosted && info.provider === "ollama") {
    notes.push("The model host did not report what this model is.");
  }
  return notes;
}
