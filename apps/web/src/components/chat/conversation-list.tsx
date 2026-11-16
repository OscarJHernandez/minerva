"use client";

import { useState } from "react";
import type { ConversationSummary } from "@/lib/chat-types";
import { splitMath } from "@/lib/inline-math";
import { RichText } from "./rich-text";

export function ConversationList({
  items,
  activeId,
  onOpen,
  onNew,
  onDelete,
  onRename,
}: {
  items: ConversationSummary[];
  activeId: string | null;
  onOpen: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  onRename?: (id: string, title: string) => void;
}) {
  const [query, setQuery] = useState("");
  const visible = items.filter((c) => c.title.toLowerCase().includes(query.trim().toLowerCase()));
  return (
    <nav aria-label="Conversations" className="flex h-full flex-col">
      <button
        type="button"
        onClick={onNew}
        className="mb-3 rounded-lg border border-rule px-3 py-2 text-left text-sm font-medium hover:bg-sunk"
      >
        + New conversation
      </button>
      <label className="mb-2 block text-xs text-muted">
        Search recent conversations
        <input
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          className="mt-1 w-full rounded border border-rule bg-raised px-2 py-1 text-sm"
        />
      </label>
      <ul className="-mx-1 flex-1 space-y-0.5 overflow-y-auto px-1">
        {visible.map((c) => (
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
              <RichText segments={splitMath(c.title)} />
            </button>
            {onRename && (
              <button
                type="button"
                aria-label={`Rename conversation ${c.title}`}
                className="px-2 py-2 text-xs text-muted"
                onClick={() => {
                  const title = window.prompt("Conversation title", c.title)?.trim();
                  if (title && title !== c.title) onRename(c.id, title);
                }}
              >
                Rename
              </button>
            )}
            <button
              type="button"
              onClick={() => onDelete(c.id)}
              aria-label={`Delete conversation ${c.title}`}
              className="px-2 py-2 text-xs text-muted hover:text-bad"
            >
              ✕
            </button>
          </li>
        ))}
        {visible.length === 0 && (
          <li className="px-2.5 py-1.5 text-sm text-muted">
            {items.length === 0 ? "No conversations yet." : "No matching conversations."}
          </li>
        )}
      </ul>
    </nav>
  );
}
