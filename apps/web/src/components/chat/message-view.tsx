import { Math } from "@minerva/ui/math";
import type { StoredMessage } from "@/lib/chat-types";
import { splitMath } from "@/lib/inline-math";
import { replyToMarkdown } from "@/lib/reply-markdown";
import { CardView } from "./card-view";
import { CopyButton, DebugPanel } from "./debug-panel";
import { ModelDetails } from "./model-details";
import { RichText } from "./rich-text";

/** A reply that asserts mathematics although no tool ran: its formulas are the model's own. */
function noToolRan(content: StoredMessage["content"]): boolean {
  if (!content.trace || content.trace.length > 0) return false; // older messages carry no trace
  if (content.stop_reason && content.stop_reason !== "completed") return false;
  return (content.blocks ?? []).some(
    (b) => b.kind === "text" && b.segments.some((s) => s.type !== "text"),
  );
}

export function MessageView({
  message,
  getReport,
  onRetry,
  retryLabel = "Regenerate",
  onEdit,
}: {
  message: StoredMessage;
  /** Answer the last message again (the reply replaces this one). Only the last message gets it. */
  onRetry?: () => void;
  retryLabel?: string;
  /** Put this message back in the box to change it. Only the last user message gets it. */
  onEdit?: () => void;
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
        {(onEdit || onRetry) && (
          <div className="flex gap-3 text-xs">
            {onEdit && (
              <button type="button" onClick={onEdit} className="text-accent hover:underline">
                Edit
              </button>
            )}
            {onRetry && (
              <button type="button" onClick={onRetry} className="text-accent hover:underline">
                Retry
              </button>
            )}
          </div>
        )}
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
      {message.content.stop_reason === "stopped" && (
        <p role="status" className="text-xs text-warn" data-testid="stopped">
          Stopped before the reply was complete.
        </p>
      )}
      {noToolRan(message.content) && (
        <p className="text-xs text-muted" data-testid="no-tool">
          No tool was run for this answer: nothing in it was computed or checked by Minerva&apos;s
          engine, and its formulas are the model&apos;s own.
        </p>
      )}
      {message.content.model_info ? (
        <ModelDetails
          info={message.content.model_info}
          suffix={
            message.content.model_info.thinking_requested == null
              ? ""
              : message.content.model_info.thinking_requested
                ? " · thinking on"
                : " · thinking off"
          }
        />
      ) : (
        message.content.debug?.provider === "openrouter" && (
          <p className="text-xs text-muted">via OpenRouter · {message.content.debug.model}</p>
        )
      )}
      {(blocks.length > 0 || onRetry) && (
        <div className="flex gap-3">
          <CopyButton label="Copy reply" getText={() => replyToMarkdown(message.content)} />
          {onRetry && (
            <button type="button" onClick={onRetry} className="text-xs text-accent hover:underline">
              {retryLabel}
            </button>
          )}
        </div>
      )}
      {getReport && <DebugPanel content={message.content} getReport={getReport} />}
    </div>
  );
}
