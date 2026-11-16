"use client";

import { useState } from "react";
import { MathField } from "./math-field";

/** A MathLive editor to compose a formula and insert it into the message as `$...$`. */
export function FormulaPopover({
  onInsert,
  onClose,
  draftKey,
}: {
  onInsert: (latex: string) => void;
  onClose: () => void;
  draftKey?: string;
}) {
  const [initial] = useState(() => {
    try {
      return draftKey ? (sessionStorage.getItem(draftKey) ?? "") : "";
    } catch {
      return "";
    }
  });
  const [latex, setLatex] = useState(initial);
  function update(value: string) {
    setLatex(value);
    if (!draftKey) return;
    try {
      if (value) sessionStorage.setItem(draftKey, value);
      else sessionStorage.removeItem(draftKey);
    } catch {
      /* Editing works without storage. */
    }
  }
  function insert(value: string) {
    update("");
    onInsert(value);
  }

  return (
    <div
      className="mb-2 rounded-lg border border-rule bg-raised p-3 shadow-sm"
      role="dialog"
      aria-label="Formula editor"
    >
      <div className="mb-2 flex items-center justify-between text-xs text-muted">
        <span>Type a formula; Enter inserts it into your message</span>
        {latex && (
          <button
            type="button"
            onClick={() => {
              update("");
              onClose();
            }}
          >
            Discard
          </button>
        )}
        <button type="button" onClick={onClose} className="hover:text-ink">
          Close
        </button>
      </div>
      <MathField
        initial={initial}
        label="Formula"
        onChange={update}
        onEnter={insert}
        onEscape={onClose}
      />
      <div className="mt-2 flex justify-end">
        <button
          type="button"
          disabled={!latex.trim()}
          onClick={() => insert(latex)}
          className="rounded bg-accent px-3 py-1.5 text-sm font-medium text-accent-ink disabled:opacity-50"
        >
          Insert
        </button>
      </div>
    </div>
  );
}
