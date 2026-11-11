"use client";

import { useCallback, useRef, useState } from "react";
import { FormulaPopover } from "./formula-popover";

interface Props {
  disabled: boolean;
  busy: boolean;
  onSend: (text: string) => void;
  onStop: () => void;
}

export function Composer({ disabled, busy, onSend, onStop }: Props) {
  const [text, setText] = useState("");
  const [formula, setFormula] = useState(false);
  const areaRef = useRef<HTMLTextAreaElement | null>(null);

  const insert = useCallback(
    (latex: string) => {
      const el = areaRef.current;
      const snippet = `$${latex.trim()}$`;
      const start = el?.selectionStart ?? text.length;
      const end = el?.selectionEnd ?? text.length;
      setText((t) => t.slice(0, start) + snippet + t.slice(end));
      setFormula(false);
      requestAnimationFrame(() => el?.focus());
    },
    [text.length],
  );
  const close = useCallback(() => setFormula(false), []);

  function submit() {
    const value = text.trim();
    if (!value || disabled || busy) return;
    onSend(value);
    setText("");
  }

  return (
    <div className="border-t border-rule bg-paper pb-4 pt-3">
      {formula && <FormulaPopover onInsert={insert} onClose={close} />}
      <div className="flex items-end gap-2 rounded-2xl border border-rule bg-raised p-2 focus-within:outline focus-within:outline-2 focus-within:outline-accent">
        <button
          type="button"
          onClick={() => setFormula((f) => !f)}
          aria-pressed={formula}
          title="Insert a formula"
          className="h-9 shrink-0 rounded-lg px-2.5 text-sm text-muted hover:bg-sunk hover:text-ink"
        >
          ∑ Formula
        </button>
        <textarea
          ref={areaRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
          rows={1}
          disabled={disabled}
          placeholder="Describe a problem, or paste a formula between $ signs…"
          aria-label="Message"
          className="max-h-40 min-h-9 flex-1 resize-none bg-transparent px-1 py-1.5 text-[15px] outline-none"
        />
        {busy ? (
          <button
            type="button"
            onClick={onStop}
            className="h-9 shrink-0 rounded-lg border border-rule px-3 text-sm hover:bg-sunk"
          >
            Stop
          </button>
        ) : (
          <button
            type="button"
            onClick={submit}
            disabled={disabled || !text.trim()}
            className="h-9 shrink-0 rounded-lg bg-accent px-4 text-sm font-medium text-accent-ink disabled:opacity-40"
          >
            Send
          </button>
        )}
      </div>
      <p className="mt-1.5 text-center text-xs text-muted">
        Results come from verified computation. Formulas written by the model alone are marked.
      </p>
    </div>
  );
}
