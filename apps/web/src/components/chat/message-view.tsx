import type { StoredMessage } from "@/lib/chat-types";
import { parseInlineMath } from "@/lib/inline-math";
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
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-sm bg-accent-soft px-4 py-2.5 text-ink">
          <RichText segments={parseInlineMath(message.content.text)} />
        </div>
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
      {getReport && <DebugPanel content={message.content} getReport={getReport} />}
    </div>
  );
}
