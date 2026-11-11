"use client";

import type { ConversationSummary } from "@/lib/chat-types";
import { parseInlineMath } from "@/lib/inline-math";
import { RichText } from "./rich-text";

export function ConversationList({
  items,
  activeId,
  onOpen,
  onNew,
  onDelete,
}: {
  items: ConversationSummary[];
  activeId: string | null;
  onOpen: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
}) {
  return (
    <nav aria-label="Conversations" className="flex h-full flex-col">
      <button
        type="button"
        onClick={onNew}
        className="mb-3 rounded-lg border border-rule px-3 py-2 text-left text-sm font-medium hover:bg-sunk"
      >
        + New conversation
      </button>
      <ul className="-mx-1 flex-1 space-y-0.5 overflow-y-auto px-1">
        {items.map((c) => (
          <li key={c.id} className="group flex items-center">
            <button
              type="button"
              onClick={() => onOpen(c.id)}
              aria-current={c.id === activeId ? "true" : undefined}
              className={`min-w-0 flex-1 truncate rounded-md px-2.5 py-1.5 text-left text-sm ${
                c.id === activeId
                  ? "bg-accent-soft text-ink"
                  : "text-muted hover:bg-sunk hover:text-ink"
              }`}
            >
              <RichText segments={parseInlineMath(c.title)} />
            </button>
            <button
              type="button"
              onClick={() => onDelete(c.id)}
              aria-label={`Delete conversation ${c.title}`}
              className="px-1.5 text-xs text-muted opacity-0 hover:text-bad focus:opacity-100 group-hover:opacity-100"
            >
              ✕
            </button>
          </li>
        ))}
        {items.length === 0 && (
          <li className="px-2.5 py-1.5 text-sm text-muted">No conversations yet.</li>
        )}
      </ul>
    </nav>
  );
}
