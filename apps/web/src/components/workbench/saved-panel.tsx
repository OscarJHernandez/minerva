"use client";

import { useState } from "react";
import { Math } from "@minerva/ui/math";
import type { DerivationSummary, HistoryItem } from "@/lib/types";
import { HistoryList } from "../history-list";

interface Props {
  derivations: DerivationSummary[];
  activeId: string | null;
  earlier: HistoryItem[];
  onOpen: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  onOpenEarlier: (item: HistoryItem) => void;
  onDeleteEarlier: (item: HistoryItem) => void;
  onClearEarlier: () => void;
}

export function SavedPanel({
  derivations,
  activeId,
  earlier,
  onOpen,
  onNew,
  onDelete,
  onOpenEarlier,
  onDeleteEarlier,
  onClearEarlier,
}: Props) {
  const [search, setSearch] = useState("");
  const matches = (...parts: (string | null)[]) =>
    parts.some((part) => part?.toLowerCase().includes(search.trim().toLowerCase()));
  const listedDerivations = derivations.filter((d) => matches(d.first_input));
  const listedCalculations = earlier.filter((item) =>
    matches(item.input, item.operation, JSON.stringify(item.assumptions)),
  );
  return (
    <div className="space-y-5">
      <label className="block text-xs text-muted">
        Search recent calculations and derivations
        <input
          type="search"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="mt-1 w-full rounded border border-rule bg-raised px-2 py-1 text-sm"
        />
      </label>
      <div>
        <div className="mb-2 flex items-center justify-between">
          <h3 className="text-sm font-medium">Derivations</h3>
          <button type="button" onClick={onNew} className="text-xs text-accent hover:underline">
            New derivation
          </button>
        </div>
        {derivations.length === 0 ? (
          <p className="text-sm text-muted">Derivations you build are saved here, privately.</p>
        ) : (
          <ul className="space-y-1.5">
            {listedDerivations.map((d) => (
              <li
                key={d.id}
                className={`group flex items-start gap-2 rounded border px-2.5 py-2 ${
                  d.id === activeId ? "border-accent bg-accent-soft" : "border-rule bg-raised"
                }`}
              >
                <button
                  type="button"
                  onClick={() => onOpen(d.id)}
                  aria-label={`Open derivation of ${d.steps} step${d.steps === 1 ? "" : "s"}, starting from ${d.first_input ?? "nothing"}`}
                  className="min-w-0 flex-1 text-left"
                >
                  <div className="text-xs text-muted">
                    {d.steps} step{d.steps === 1 ? "" : "s"} ·{" "}
                    {new Date(d.updated_at).toLocaleDateString()}
                  </div>
                  <div className="overflow-hidden text-sm">
                    {d.first_input &&
                      (d.first_format === "latex" ? (
                        <Math latex={d.first_input} />
                      ) : (
                        <code className="font-mono text-xs">{d.first_input}</code>
                      ))}
                  </div>
                </button>
                <button
                  type="button"
                  onClick={() => onDelete(d.id)}
                  aria-label="Delete this derivation"
                  className="px-2 py-1 text-xs text-muted hover:text-bad"
                >
                  ✕
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
      {earlier.length > 0 && (
        <div>
          <HistoryList
            items={listedCalculations}
            activeId={null}
            onOpen={onOpenEarlier}
            onDelete={onDeleteEarlier}
            onClear={onClearEarlier}
          />
        </div>
      )}
    </div>
  );
}
