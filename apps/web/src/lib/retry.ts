import type { EquationReference, StoredMessage, UserEquation } from "./chat-types";

/** The conversation up to and including its last user message, whose text can be replaced. */
export function untilLastUser(messages: StoredMessage[], text?: string): StoredMessage[] {
  const last = messages.map((m) => m.role).lastIndexOf("user");
  if (last < 0) return messages;
  const kept = messages.slice(0, last + 1);
  if (text === undefined) return kept;
  const user = kept[last]!;
  return [...kept.slice(0, last), { ...user, content: { text } }];
}

/** Why a reply ended early: these are worth saying "Retry" for, not "Regenerate". */
const INCOMPLETE = new Set([
  "model_error",
  "malformed_output",
  "internal_error",
  "time_limit",
  "step_limit",
  "tool_call_limit",
  "repeated_calls",
  "output_limit",
  "stopped",
]);

/** "Retry" when the reply ended early or failed, "Regenerate" when it is complete and the user wants another. */
export function retryLabel(stopReason: string | null | undefined): "Retry" | "Regenerate" {
  return INCOMPLETE.has(stopReason ?? "") ? "Retry" : "Regenerate";
}

/** Adds what the server found in the last user message (its equations, its references) to the one on screen. */
export function withUserDetails(
  messages: StoredMessage[],
  details: { equations?: UserEquation[]; references?: EquationReference[] },
): StoredMessage[] {
  const last = messages.map((m) => m.role).lastIndexOf("user");
  const user = messages[last];
  if (!user) return messages;
  const content = {
    ...user.content,
    ...(details.equations?.length ? { equations: details.equations } : {}),
    ...(details.references?.length ? { references: details.references } : {}),
  };
  return [...messages.slice(0, last), { ...user, content }, ...messages.slice(last + 1)];
}
