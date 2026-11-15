import { Math } from "@minerva/ui/math";
import type { StoredMessage } from "@/lib/chat-types";
import { splitMath } from "@/lib/inline-math";
import { CardView } from "./card-view";
import { DebugPanel } from "./debug-panel";
import { RichText } from "./rich-text";

export function MessageView({
  message,
  getReport,
}: {
  message: StoredMessage;
  /** Present only in debug mode: builds the copyable report for this turn. */
  getReport?: () => string;
}) {
  if (message.role === "user") {
    const numbers = (message.content.equations ?? []).map((e) => e.number);
    let next = 0;
    // The server numbers the displayed formulas of a message in order; the same splitter reads them here.
    const segments = splitMath(message.content.text).map((s) =>
      s.type === "display_math" ? { ...s, number: numbers[next++] ?? null } : s,
    );
    const references = message.content.references ?? [];
    return (
      <div className="flex flex-col items-end gap-1">
        <div className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-sm bg-accent-soft px-4 py-2.5 text-ink">
          <RichText segments={segments} />
        </div>
        {references.length > 0 && (
          <ul
            className="max-w-[85%] space-y-0.5 text-right text-xs text-muted"
            data-testid="equation-references"
          >
            {references.map((r) => (
              <li key={r.number}>
                {r.latex === null ? (
                  <>Equation ({r.number}): there is no such equation in this conversation.</>
                ) : (
                  <>
                    Taken as equation ({r.number}): <Math latex={r.latex} />
                    {r.source === "written_by_model" && " (written by the model, not verified)"}
                  </>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>
    );
  }
  const { blocks = [] } = message.content;
  return (
    <div className="space-y-3">
      {blocks.map((block, i) =>
        block.kind === "card" ? (
          <CardView key={block.card.id} card={block.card} />
        ) : (
          <RichText key={i} segments={block.segments} />
        ),
      )}
      {(message.content.context_trimmed ?? 0) > 0 && (
        <p role="status" className="text-xs text-warn" data-testid="context-trimmed">
          Earlier messages were left out of this answer because they did not fit the model&apos;s
          context window. Start a new conversation, or ask the operator for a larger window.
        </p>
      )}
      {message.content.debug?.provider === "openrouter" && (
        <p className="text-xs text-muted">via OpenRouter · {message.content.debug.model}</p>
      )}
      {getReport && <DebugPanel content={message.content} getReport={getReport} />}
    </div>
  );
}
