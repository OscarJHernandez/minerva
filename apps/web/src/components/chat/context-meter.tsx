"use client";

import type { StoredMessage } from "@/lib/chat-types";
import { conversationUsage, level, type TokenSums } from "@/lib/usage";

const n = (v: number) => v.toLocaleString("en-US");

function sumText(s: TokenSums): string {
  const note = s.unreported > 0 ? ` (${s.unreported} call(s) reported no count)` : "";
  return `${n(s.prompt)} in, ${n(s.completion)} out${note}`;
}

/**
 * Always-visible context use: the window, what the last request used (the provider's count, or the
 * server's estimate, labelled), a warning near the limit or when earlier messages were left out, and
 * token totals. Counts come from the debug data stored with each reply; nothing here is computed
 * from message text and a count the provider did not report is left out, never made up.
 */
export function ContextMeter({ messages }: { messages: StoredMessage[] }) {
  const u = conversationUsage(messages);
  if (!u) return null;
  const lv = level(u.used, u.window);
  const pct =
    u.window && u.used !== null ? Math.min(100, Math.round((u.used / u.window) * 100)) : 0;
  const bar = lv === "full" ? "bg-bad" : lv === "near" ? "bg-warn" : "bg-ok";
  const label =
    u.used === null
      ? "Context use not reported"
      : u.window
        ? `${n(u.used)} / ${n(u.window)} tokens${u.estimated ? " (estimate)" : ""}`
        : `${n(u.used)} prompt tokens${u.estimated ? " (estimate)" : ""}`;
  const warning =
    u.trimmed > 0
      ? `${u.trimmed} earlier message${u.trimmed === 1 ? " was" : "s were"} left out of the last reply to fit the model's window; it cannot see them.`
      : lv === "full"
        ? "The last request filled the model's window: older messages may be cut off."
        : lv === "near"
          ? "The conversation is close to the model's window. Start a new conversation soon."
          : null;
  return (
    <div className="mb-1 text-xs text-muted" data-testid="context-meter">
      <details>
        <summary className="flex cursor-pointer items-center gap-2" aria-label="Context use">
          {u.window && u.used !== null && (
            <span
              role="meter"
              aria-label="Context window used"
              aria-valuemin={0}
              aria-valuemax={u.window}
              aria-valuenow={Math.min(u.used, u.window)}
              className="h-1.5 w-20 shrink-0 overflow-hidden rounded-full bg-sunk"
            >
              <span className={`block h-full ${bar}`} style={{ width: `${pct}%` }} />
            </span>
          )}
          <span data-testid="context-label">{label}</span>
        </summary>
        <dl className="mt-1 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 rounded bg-sunk px-2 py-1.5">
          {u.lastTurn && (
            <>
              <dt>Last reply</dt>
              <dd>{sumText(u.lastTurn)}</dd>
            </>
          )}
          <dt>
            Conversation ({u.turns} repl{u.turns === 1 ? "y" : "ies"})
          </dt>
          <dd>{sumText(u.total)}</dd>
          <dt>Source</dt>
          <dd>
            {u.estimated
              ? "Minerva's estimate: the provider reported no count."
              : "As the provider reported it."}
          </dd>
        </dl>
      </details>
      {warning && (
        <p
          role="alert"
          className={`mt-1 rounded px-2 py-1 ${lv === "full" ? "bg-bad-soft text-bad" : "bg-warn-soft text-warn"}`}
          data-testid="context-usage-warning"
        >
          {warning}
        </p>
      )}
    </div>
  );
}
