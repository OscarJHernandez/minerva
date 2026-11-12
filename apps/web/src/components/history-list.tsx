"use client";

import { Math } from "@minerva/ui/math";
import type { HistoryItem } from "@/lib/types";

const DOT: Record<string, string> = {
  ok: "bg-ok",
  unchanged: "bg-muted",
  parse_error: "bg-bad",
  internal_error: "bg-bad",
};

const OPERATION_LABEL: Record<string, string> = {
  simplify: "Simplify",
  differentiate: "Differentiate",
  identities: "Identities",
};

export function HistoryList({
  items,
  activeId,
  onOpen,
  onDelete,
  onClear,
}: {
  items: HistoryItem[];
  activeId: string | null;
  onOpen: (item: HistoryItem) => void;
  onDelete: (item: HistoryItem) => void;
  onClear: () => void;
}) {
  return (
    <section aria-labelledby="history-heading">
      <div className="mb-2 flex items-center justify-between">
        <h2 id="history-heading" className="text-sm font-medium">
          History
        </h2>
        {items.length > 0 && (
          <button type="button" onClick={onClear} className="text-xs text-muted hover:text-bad">
            Clear all
          </button>
        )}
      </div>
      {items.length === 0 ? (
        <p className="text-sm text-muted">
          Saved calculations appear here. They are private to you.
        </p>
      ) : (
        <ul className="space-y-1.5">
          {items.map((item) => (
            <li
              key={item.id}
              className={`group flex items-start gap-2 rounded border px-2.5 py-2 ${
                item.id === activeId ? "border-accent bg-accent-soft" : "border-rule bg-raised"
              }`}
            >
              <button
                type="button"
                onClick={() => onOpen(item)}
                className="min-w-0 flex-1 text-left"
                aria-label={`Open ${item.operation}: ${item.input}`}
              >
                <div className="flex items-center gap-1.5 text-xs text-muted">
                  <span className={`h-1.5 w-1.5 rounded-full ${DOT[item.status] ?? "bg-warn"}`} />
                  {OPERATION_LABEL[item.operation] ?? "Identities"}
                </div>
                <div className="overflow-hidden text-sm">
                  {item.format === "latex" ? (
                    <Math latex={item.input} />
                  ) : (
                    <code className="font-mono text-xs">{item.input}</code>
                  )}
                </div>
              </button>
              <button
                type="button"
                onClick={() => onDelete(item)}
                aria-label="Delete this calculation"
                className="text-xs text-muted opacity-0 hover:text-bad focus:opacity-100 group-hover:opacity-100"
              >
                ✕
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
