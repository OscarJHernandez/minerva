"use client";

import { Math } from "@minerva/ui/math";
import { useState } from "react";
import type { StepResult, Subterm } from "@/lib/types";

export type PartAction = "expand" | "factor" | "cancel" | "simplify";

interface Props {
  /** The parts listed for the current expression, if any. */
  result: StepResult | null;
  /** The expression changed since the list was made. */
  stale: boolean;
  searching: boolean;
  chosen: Subterm | null;
  disabled: boolean;
  onList: (find: string) => void;
  onChoose: (part: Subterm | null) => void;
  onRun: (action: PartAction) => void;
}

const ACTIONS: { id: PartAction; label: string }[] = [
  { id: "expand", label: "Expand" },
  { id: "factor", label: "Factor" },
  { id: "cancel", label: "Cancel" },
  { id: "simplify", label: "Simplify" },
];

/**
 * Choosing one part of the expression to change. The server lists the parts (it alone reads the
 * expression); the choice is made from that list, and a choice that no longer matches the expression is
 * refused rather than applied somewhere else. Collect and identities then act on the chosen part.
 */
export function PartPicker({
  result,
  stale,
  searching,
  chosen,
  disabled,
  onList,
  onChoose,
  onRun,
}: Props) {
  const [find, setFind] = useState("");
  const parts = result?.subterms ?? [];
  return (
    <div
      aria-label="Choose a part"
      className="mt-3 space-y-3 rounded-lg border border-rule bg-sunk px-3 py-2.5"
    >
      <form
        className="flex flex-wrap items-center gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          onList(find.trim());
        }}
      >
        <label className="flex items-center gap-1.5 text-xs text-muted">
          Look for
          <input
            value={find}
            onChange={(e) => setFind(e.target.value)}
            aria-label="Part to look for (optional)"
            placeholder="optional, e.g. x^2 - 1"
            maxLength={400}
            className="w-48 rounded border border-rule bg-raised px-2 py-1.5 font-mono text-sm text-ink"
          />
        </label>
        <button
          type="submit"
          disabled={disabled}
          className="rounded border border-accent px-3 py-1.5 text-sm font-medium text-accent hover:bg-accent-soft disabled:opacity-50"
        >
          {searching ? "Listing…" : "List parts"}
        </button>
      </form>

      {stale && <p className="text-xs text-warn">The expression changed: list the parts again.</p>}
      {!stale && result && parts.length === 0 && (
        <p className="text-sm text-muted">{result.message}</p>
      )}
      {!stale && parts.length > 0 && (
        <ul aria-label="Parts" className="max-h-64 space-y-1 overflow-y-auto">
          {parts.map((part) => {
            const on = chosen?.index === part.index;
            return (
              <li key={part.index}>
                <button
                  type="button"
                  data-testid={`part-${part.index}`}
                  aria-label={`Part ${part.index}: ${part.text}${
                    part.occurrences > 1
                      ? `, occurrence ${part.occurrence} of ${part.occurrences}`
                      : ""
                  }`}
                  aria-pressed={on}
                  onClick={() => onChoose(on ? null : part)}
                  className={`flex w-full items-center justify-between gap-3 overflow-x-auto rounded border px-2 py-1 text-left ${
                    on ? "border-accent bg-accent-soft" : "border-rule bg-raised hover:bg-sunk"
                  }`}
                >
                  <Math latex={part.highlight_latex} />
                  {part.occurrences > 1 && (
                    <span className="shrink-0 text-xs text-muted">
                      {part.occurrence} of {part.occurrences}
                    </span>
                  )}
                </button>
              </li>
            );
          })}
        </ul>
      )}
      {result?.truncated && !stale && (
        <p className="text-xs text-muted">More parts exist: type the part to narrow the list.</p>
      )}

      <div className="flex flex-wrap items-center gap-2">
        {chosen ? (
          <span className="text-sm" data-testid="chosen-part">
            Changing: <Math latex={chosen.latex} />{" "}
            <button
              type="button"
              onClick={() => onChoose(null)}
              className="text-xs text-accent hover:underline"
            >
              Clear
            </button>
          </span>
        ) : (
          <span className="text-xs text-muted">Choose a part above.</span>
        )}
        <span className="ml-auto flex flex-wrap gap-1.5">
          {ACTIONS.map((a) => (
            <button
              key={a.id}
              type="button"
              disabled={disabled || !chosen}
              onClick={() => onRun(a.id)}
              className="rounded border border-rule px-3 py-1.5 text-sm font-medium text-ink hover:bg-raised disabled:opacity-50"
            >
              {a.label}
            </button>
          ))}
        </span>
      </div>
    </div>
  );
}
