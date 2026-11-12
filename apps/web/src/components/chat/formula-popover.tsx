"use client";

import { useState } from "react";
import { MathField } from "./math-field";

/** A MathLive editor to compose a formula and insert it into the message as `$...$`. */
export function FormulaPopover({
  onInsert,
  onClose,
}: {
  onInsert: (latex: string) => void;
  onClose: () => void;
}) {
  const [latex, setLatex] = useState("");

  return (
    <div
      className="mb-2 rounded-lg border border-rule bg-raised p-3 shadow-sm"
      role="dialog"
      aria-label="Formula editor"
    >
      <div className="mb-2 flex items-center justify-between text-xs text-muted">
        <span>Type a formula; Enter inserts it into your message</span>
        <button type="button" onClick={onClose} className="hover:text-ink">
          Close
        </button>
      </div>
      <MathField label="Formula" onChange={setLatex} onEnter={onInsert} onEscape={onClose} />
      <div className="mt-2 flex justify-end">
        <button
          type="button"
          disabled={!latex.trim()}
          onClick={() => onInsert(latex)}
          className="rounded bg-accent px-3 py-1.5 text-sm font-medium text-accent-ink disabled:opacity-50"
        >
          Insert
        </button>
      </div>
    </div>
  );
}
