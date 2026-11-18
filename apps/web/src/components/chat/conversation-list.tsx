"use client";

import { useEffect, useRef, useState } from "react";
import type { ConversationSummary } from "@/lib/chat-types";
import { splitMath } from "@/lib/inline-math";
import { RichText } from "./rich-text";

/** A row's actions take no room until they are wanted: on hover, on keyboard focus, and always on the open
 * conversation (a touch screen has no hover, and tapping a row opens it). */
function revealed(active: boolean): string {
  return active
    ? "opacity-100"
    : "opacity-0 group-hover:opacity-100 group-focus-within:opacity-100 focus-visible:opacity-100";
}

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
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  // Set once an edit has ended (Enter, Escape, blur) so the blur that follows cannot save a second time.
  const closed = useRef(true);
  const visible = items.filter((c) => c.title.toLowerCase().includes(query.trim().toLowerCase()));

  useEffect(() => {
    if (editing) {
      inputRef.current?.focus();
      inputRef.current?.select();
    }
  }, [editing]);

  function startEditing(c: ConversationSummary) {
    if (!onRename) return;
    closed.current = false;
    setDraft(c.title);
    setEditing(c.id);
  }

  function finishEditing(c: ConversationSummary, save: boolean) {
    if (closed.current) return;
    closed.current = true;
    setEditing(null);
    const title = draft.trim();
    if (save && title && title !== c.title) onRename?.(c.id, title);
  }

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
        {visible.map((c) => {
          const active = c.id === activeId;
          return (
            <li key={c.id} className="group flex items-center">
              {editing === c.id ? (
                <input
                  ref={inputRef}
                  aria-label={`Title of conversation ${c.title}`}
                  value={draft}
                  maxLength={200}
                  onChange={(event) => setDraft(event.target.value)}
                  onBlur={() => finishEditing(c, true)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") finishEditing(c, true);
                    if (event.key === "Escape") finishEditing(c, false);
                  }}
                  className="min-w-0 flex-1 rounded-md border border-accent bg-raised px-2.5 py-1.5 text-sm text-ink"
                />
              ) : (
                <>
                  <button
                    type="button"
                    onClick={() => onOpen(c.id)}
                    onDoubleClick={() => startEditing(c)}
                    onKeyDown={(event) => {
                      if (event.key === "F2") startEditing(c);
                    }}
                    aria-current={active ? "true" : undefined}
                    title={onRename ? "Double-click or press F2 to rename" : undefined}
                    className={`min-w-0 flex-1 truncate rounded-md px-2.5 py-1.5 text-left text-sm ${
                      active ? "bg-accent-soft text-ink" : "text-muted hover:bg-sunk hover:text-ink"
                    }`}
                  >
                    <RichText segments={splitMath(c.title)} />
                  </button>
                  {onRename && (
                    <button
                      type="button"
                      onClick={() => startEditing(c)}
                      aria-label={`Rename conversation ${c.title}`}
                      className={`px-1.5 py-2 text-xs text-muted hover:text-ink ${revealed(active)}`}
                    >
                      ✎
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={() => onDelete(c.id)}
                    aria-label={`Delete conversation ${c.title}`}
                    className={`px-1.5 py-2 text-xs text-muted hover:text-bad ${revealed(active)}`}
                  >
                    ✕
                  </button>
                </>
              )}
            </li>
          );
        })}
        {visible.length === 0 && (
          <li className="px-2.5 py-1.5 text-sm text-muted">
            {items.length === 0 ? "No conversations yet." : "No matching conversations."}
          </li>
        )}
      </ul>
    </nav>
  );
}
