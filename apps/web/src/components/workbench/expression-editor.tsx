"use client";

import { Math } from "@minerva/ui/math";
import { useEffect, useRef } from "react";
import { MathInput } from "../math-input";

export type EditorFormat = "latex" | "text";

export interface Continuing {
  /** 1-based position of the step being continued. */
  step: number;
  /** The step's result, for display only. */
  latex: string | null;
  /** The text is still exactly that step's result, so the new step is a true continuation. */
  intact: boolean;
}

interface Props {
  format: EditorFormat;
  latex: string;
  text: string;
  onLatex: (value: string) => void;
  onText: (value: string) => void;
  onFormat: (format: EditorFormat) => void;
  onSubmit: () => void;
  continuing: Continuing | null;
  onClear: () => void;
  /** Bumped to move the cursor into the plain-text editor (the "Edit" action on a step). */
  focusSignal: number;
}

/**
 * The expression being worked on. Either the math editor (LaTeX) or plain text. A step is carried
 * forward as text, because that is the form the server reads back exactly; the LaTeX shown beside
 * a result is for reading only.
 */
export function ExpressionEditor({
  format,
  latex,
  text,
  onLatex,
  onText,
  onFormat,
  onSubmit,
  continuing,
  onClear,
  focusSignal,
}: Props) {
  const area = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    if (focusSignal > 0) area.current?.focus();
  }, [focusSignal]);

  const hasValue = (format === "latex" ? latex : text).trim() !== "";

  return (
    <section aria-label="Expression">
      {format === "latex" ? (
        <MathInput value={latex} onChange={onLatex} onSubmit={onSubmit} />
      ) : (
        <div>
          <label htmlFor="expr-text" className="mb-1.5 block text-sm font-medium">
            Expression (text syntax)
          </label>
          <textarea
            id="expr-text"
            ref={area}
            value={text}
            onChange={(e) => onText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                onSubmit();
              }
            }}
            rows={2}
            spellCheck={false}
            className="w-full rounded border border-rule bg-raised p-3 font-mono text-sm"
            placeholder="sin(x)**2 + cos(x)**2"
          />
        </div>
      )}

      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted">
        <button
          type="button"
          onClick={() => onFormat(format === "latex" ? "text" : "latex")}
          className="text-accent hover:underline"
        >
          {format === "latex" ? "Edit as plain text" : "Use the math editor"}
        </button>
        {hasValue && (
          <button type="button" onClick={onClear} className="hover:text-ink hover:underline">
            Clear
          </button>
        )}
        {format === "text" && (
          <span>Text syntax: sin(x)**2, sqrt(x), exp(x), pi. E, I, pi and oo are constants.</span>
        )}
      </div>

      {continuing && (
        <div
          role="status"
          className={`mt-3 rounded border px-3 py-2 text-sm ${
            continuing.intact ? "border-rule bg-sunk" : "border-rule bg-warn-soft text-warn"
          }`}
        >
          {continuing.intact ? (
            <>
              <span className="text-muted">
                Continuing from the result of step {continuing.step}
              </span>
              {continuing.latex && <Math latex={continuing.latex} display />}
            </>
          ) : (
            <>
              Edited: the next step will be a new input, not a continuation of step{" "}
              {continuing.step}.
            </>
          )}
        </div>
      )}
    </section>
  );
}
