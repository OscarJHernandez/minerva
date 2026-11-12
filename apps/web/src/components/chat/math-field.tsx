"use client";

import { useEffect, useRef, useState } from "react";
import type { MathfieldElement } from "mathlive";

interface Props {
  /** Initial LaTeX; later changes come from the user typing. */
  initial?: string;
  label: string;
  onChange: (latex: string) => void;
  onEnter?: (latex: string) => void;
  onEscape?: () => void;
}

/** A MathLive editor (lazy-loaded) that reports LaTeX. Shared by the formula popover and the image panel. */
export function MathField({ initial = "", label, onChange, onEnter, onEscape }: Props) {
  const ref = useRef<MathfieldElement | null>(null);
  const [ready, setReady] = useState(false);
  const handlers = useRef({ onChange, onEnter, onEscape });
  useEffect(() => {
    handlers.current = { onChange, onEnter, onEscape };
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
    if (!ready || !el) return;
    el.mathVirtualKeyboardPolicy = "auto";
    if (initial) el.setValue(initial, { silenceNotifications: true });
    el.focus();
    const onInput = () => handlers.current.onChange(el.value);
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Enter") {
        event.preventDefault();
        if (el.value.trim()) handlers.current.onEnter?.(el.value);
      }
      if (event.key === "Escape") handlers.current.onEscape?.();
    };
    el.addEventListener("input", onInput);
    el.addEventListener("keydown", onKey);
    return () => {
      el.removeEventListener("input", onInput);
      el.removeEventListener("keydown", onKey);
    };
    // `initial` is read once when the editor is ready.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready]);

  return <math-field ref={ref} aria-label={label} />;
}
