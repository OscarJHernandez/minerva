"use client";

import { useEffect, useRef, useState } from "react";
import type { MathfieldElement } from "mathlive";

/** A MathLive editor to compose a formula and insert it into the message as `$...$`. */
export function FormulaPopover({
  onInsert,
  onClose,
}: {
  onInsert: (latex: string) => void;
  onClose: () => void;
}) {
  const ref = useRef<MathfieldElement | null>(null);
  const [ready, setReady] = useState(false);
  const [latex, setLatex] = useState("");

  useEffect(() => {
    let cancelled = false;
    import("mathlive").then(({ MathfieldElement }) => {
      if (cancelled) return;
      MathfieldElement.fontsDirectory = null;
      MathfieldElement.soundsDirectory = null;
      setReady(true);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const el = ref.current;
    if (!ready || !el) return;
    el.mathVirtualKeyboardPolicy = "auto";
    el.focus();
    const onInput = () => setLatex(el.value);
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Enter") {
        event.preventDefault();
        if (el.value.trim()) onInsert(el.value);
      }
      if (event.key === "Escape") onClose();
    };
    el.addEventListener("input", onInput);
    el.addEventListener("keydown", onKey);
    return () => {
      el.removeEventListener("input", onInput);
      el.removeEventListener("keydown", onKey);
    };
  }, [ready, onInsert, onClose]);

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
      <math-field ref={ref} aria-label="Formula" />
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
