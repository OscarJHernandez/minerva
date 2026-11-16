"use client";

import { Math } from "@minerva/ui/math";
import { useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import type { Assumptions } from "@/lib/types";
import { MathInput } from "../math-input";

export type EditorFormat = "latex" | "text";

export interface Continuing {
  /** 1-based position of the step being continued. */
  step: number;
  /** The step's result, for display only. */
  latex: string | null;
  /** The text is still exactly that step's result, so the new step is a true continuation. */
  intact: boolean;
  taylor?: boolean;
  domain?: { exclusions: string[]; incomplete: boolean } | null;
}

interface Props {
  format: EditorFormat;
  latex: string;
  text: string;
  onLatex: (value: string) => void;
  onText: (value: string) => void;
  onFormat: (format: EditorFormat) => void;
  onSubmit: () => void;
  assumptions?: Assumptions;
  onConvert?: (format: EditorFormat, value: string) => void;
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
  assumptions,
  onConvert,
  continuing,
  onClear,
  focusSignal,
}: Props) {
  const area = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    if (focusSignal > 0) area.current?.focus();
  }, [focusSignal]);

  const value = format === "latex" ? latex : text;
  const hasValue = value.trim() !== "";
  const [feedback, setFeedback] = useState<{
    key: string;
    message: string;
    readAs?: string | null;
  } | null>(null);
  const inputKey = JSON.stringify([format, value, assumptions]);
  const conversion = useRef<AbortController | null>(null);
  useEffect(() => {
    if (!hasValue || !assumptions) return;
    const controller = new AbortController();
    const timer = setTimeout(() => {
      void api.inspect({ input: value, format, assumptions }, controller.signal).then(
        (result) => {
          if (!controller.signal.aborted) {
            setFeedback((old) =>
              old?.key === inputKey && old.message
                ? { ...old, readAs: result.read_as }
                : { key: inputKey, message: result.message, readAs: result.read_as },
            );
          }
        },
        () => {
          /* Typing and running a calculation still work offline. */
        },
      );
    }, 450);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [hasValue, assumptions, format, value, inputKey]);
  useEffect(() => () => conversion.current?.abort(), [inputKey]);

  const otherFormat = format === "latex" ? "text" : "latex";
  const otherValue = format === "latex" ? text : latex;

  async function switchFormat() {
    if (!hasValue || !onConvert || !assumptions || otherValue.trim()) {
      onFormat(otherFormat);
      return;
    }
    conversion.current?.abort();
    const controller = new AbortController();
    conversion.current = controller;
    try {
      const result = await api.inspect(
        { input: value, format, assumptions },
        controller.signal,
        otherFormat,
      );
      if (controller.signal.aborted) return;
      if (result.converted !== null) onConvert(otherFormat, result.converted);
      else setFeedback({ key: inputKey, message: result.message, readAs: result.read_as });
    } catch {
      if (!controller.signal.aborted) {
        setFeedback({
          key: inputKey,
          message: "Could not convert the expression. Your input is still here.",
        });
      }
    }
  }

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
              if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && !e.nativeEvent.isComposing) {
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
          onClick={() => void switchFormat()}
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

      {format === "text" && (
        <p className="mt-1 text-xs text-muted">Ctrl/Cmd+Enter runs the operation.</p>
      )}
      {feedback?.key === inputKey && (
        <div className="mt-2 text-xs text-muted">
          {feedback.message && <p role="status">{feedback.message}</p>}
          {feedback.readAs && (
            <details className="overflow-x-auto">
              <summary>How Minerva reads this</summary>
              <Math latex={feedback.readAs} display />
            </details>
          )}
        </div>
      )}
      {!hasValue && otherValue.trim() && (
        <div className="mt-2 rounded border border-rule bg-sunk p-3 text-sm" role="status">
          <p>
            Your expression is still in {otherFormat === "latex" ? "LaTeX" : "plain text"}. Use the
            format switch to return to it, or enter this format yourself.
          </p>
          <div className="mt-2 overflow-x-auto">
            {otherFormat === "latex" ? (
              <Math latex={otherValue} display />
            ) : (
              <code>{otherValue}</code>
            )}
          </div>
        </div>
      )}
      {hasValue && otherValue.trim() && !onConvert && (
        <p className="mt-2 text-xs text-muted">
          LaTeX and plain text are separate drafts. Switching formats restores the expression last
          entered in that format; it does not convert it.
        </p>
      )}

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
                Continuing from {continuing.taylor ? "the Taylor polynomial in" : "the result of"}{" "}
                step {continuing.step}
              </span>
              {continuing.latex && <Math latex={continuing.latex} display />}
              {continuing.domain?.exclusions.map((exclusion) => (
                <div key={exclusion}>
                  Source restriction: <Math latex={exclusion} />
                </div>
              ))}
              {continuing.domain?.incomplete && (
                <p>Some source restrictions could not be determined.</p>
              )}
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
