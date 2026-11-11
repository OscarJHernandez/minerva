"use client";

import { useEffect, useRef, useState } from "react";
import type { MathfieldElement } from "mathlive";

interface Props {
  value: string;
  onChange: (latex: string) => void;
  onSubmit: () => void;
}

/**
 * Mathematical input. MathLive's `<math-field>` produces LaTeX; a raw LaTeX mode is
 * available for pasting formulas. Fonts come from the KaTeX stylesheet already loaded
 * globally, so MathLive's own font directory and sounds are disabled.
 */
export function MathInput({ value, onChange, onSubmit }: Props) {
  const ref = useRef<MathfieldElement | null>(null);
  const [raw, setRaw] = useState(false);
  const [ready, setReady] = useState(false);
  const submitRef = useRef(onSubmit);
  const changeRef = useRef(onChange);
  useEffect(() => {
    submitRef.current = onSubmit;
    changeRef.current = onChange;
  });

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
    if (!ready || raw || !el) return;
    el.mathVirtualKeyboardPolicy = "auto";
    el.smartFence = true;
    if (el.value !== value) el.setValue(value, { silenceNotifications: true });
    const handleInput = () => changeRef.current(el.value);
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        submitRef.current();
      }
    };
    el.addEventListener("input", handleInput);
    el.addEventListener("keydown", handleKey);
    return () => {
      el.removeEventListener("input", handleInput);
      el.removeEventListener("keydown", handleKey);
    };
    // `value` is intentionally excluded: external updates are applied below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, raw]);

  useEffect(() => {
    const el = ref.current;
    if (ready && !raw && el && el.value !== value)
      el.setValue(value, { silenceNotifications: true });
  }, [value, ready, raw]);

  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between">
        <label htmlFor="expr" className="text-sm font-medium">
          Expression
        </label>
        <button
          type="button"
          onClick={() => setRaw((r) => !r)}
          className="text-xs text-accent hover:underline"
          aria-pressed={raw}
        >
          {raw ? "Use math editor" : "Edit as LaTeX"}
        </button>
      </div>
      {raw ? (
        <textarea
          id="expr"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              onSubmit();
            }
          }}
          rows={2}
          spellCheck={false}
          className="w-full rounded border border-rule bg-raised p-3 font-mono text-sm"
          placeholder="\\sin^2 x + \\cos^2 x"
        />
      ) : (
        <math-field ref={ref} id="expr" aria-label="Mathematical expression" />
      )}
    </div>
  );
}
