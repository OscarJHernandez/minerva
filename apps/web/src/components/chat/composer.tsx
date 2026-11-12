"use client";

import { useCallback, useLayoutEffect, useRef, useState } from "react";
import { ACCEPTED_IMAGE_TYPES, checkImageFile } from "@/lib/image-prep";
import { FormulaPopover } from "./formula-popover";
import { ImageFormulasPanel } from "./image-formulas-panel";

interface Props {
  disabled: boolean;
  busy: boolean;
  onSend: (text: string) => void;
  onStop: () => void;
  /** The server can read formulas from a picture. Hides the "Add image" button when false. */
  imagesEnabled?: boolean;
}

const MAX_HEIGHT_VH = 0.4;

export function Composer({ disabled, busy, onSend, onStop, imagesEnabled = false }: Props) {
  const [text, setText] = useState("");
  const [formula, setFormula] = useState(false);
  const [image, setImage] = useState<File | null>(null);
  const [imageError, setImageError] = useState<string | null>(null);
  const areaRef = useRef<HTMLTextAreaElement | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);
  // Once the user drags the resize handle their height wins until the message is sent.
  const manualHeight = useRef(false);
  const autoHeight = useRef(0);

  // Grow with the content up to MAX_HEIGHT_VH of the viewport, then scroll inside.
  useLayoutEffect(() => {
    const el = areaRef.current;
    if (!el || manualHeight.current) return;
    el.style.height = "auto";
    const cap = Math.round(window.innerHeight * MAX_HEIGHT_VH);
    autoHeight.current = Math.min(el.scrollHeight, cap);
    el.style.height = `${autoHeight.current}px`;
  }, [text]);

  const insertMany = useCallback(
    (formulas: string[]) => {
      const el = areaRef.current;
      const snippet = formulas.map((latex) => `$${latex.trim()}$`).join(" ");
      const start = el?.selectionStart ?? text.length;
      const end = el?.selectionEnd ?? text.length;
      setText((t) => t.slice(0, start) + snippet + t.slice(end));
      setFormula(false);
      setImage(null);
      requestAnimationFrame(() => el?.focus());
    },
    [text.length],
  );
  const insert = useCallback((latex: string) => insertMany([latex]), [insertMany]);
  const close = useCallback(() => setFormula(false), []);
  const closeImage = useCallback(() => setImage(null), []);

  function addImage(file: File | undefined) {
    if (!file) return;
    const problem = checkImageFile(file);
    setImageError(problem);
    if (problem) return;
    setFormula(false);
    setImage(file);
  }

  function submit() {
    const value = text.trim();
    if (!value || disabled || busy) return;
    onSend(value);
    setText("");
    manualHeight.current = false;
    if (areaRef.current) areaRef.current.style.height = "";
  }

  return (
    <div className="border-t border-rule bg-paper pb-4 pt-3">
      {formula && <FormulaPopover onInsert={insert} onClose={close} />}
      {image && <ImageFormulasPanel file={image} onInsert={insertMany} onClose={closeImage} />}
      {imageError && (
        <p role="alert" className="mb-2 rounded bg-bad-soft px-3 py-2 text-sm text-bad">
          {imageError}
        </p>
      )}
      <div className="rounded-2xl border border-rule bg-raised focus-within:outline focus-within:outline-2 focus-within:outline-accent">
        <textarea
          ref={areaRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onPointerUp={(e) => {
            // The browser's resize handle changes the height without an input event.
            if (Math.abs(e.currentTarget.offsetHeight - autoHeight.current) > 1) {
              manualHeight.current = true;
            }
          }}
          onPaste={(e) => {
            const file = imagesEnabled
              ? Array.from(e.clipboardData.files).find((f) => f.type.startsWith("image/"))
              : undefined;
            if (!file) return; // plain text pastes as usual
            e.preventDefault();
            addImage(file);
          }}
          onKeyDown={(e) => {
            // Enter writes a new line; Ctrl+Enter or Cmd+Enter sends.
            if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && !e.nativeEvent.isComposing) {
              e.preventDefault();
              submit();
            }
          }}
          rows={3}
          disabled={disabled}
          placeholder="Describe a problem, or paste a formula between $ signs…"
          aria-label="Message"
          aria-keyshortcuts="Control+Enter Meta+Enter"
          className="block max-h-[40vh] min-h-[5.5rem] w-full resize-y bg-transparent px-3.5 pb-1 pt-3 text-[15px] leading-6 outline-none"
        />
        <div className="flex items-center gap-1 px-2 pb-2">
          <button
            type="button"
            onClick={() => setFormula((f) => !f)}
            aria-pressed={formula}
            title="Insert a formula"
            aria-label="Insert a formula"
            className="h-9 shrink-0 rounded-lg px-2.5 text-sm text-muted hover:bg-sunk hover:text-ink"
          >
            ∑<span className="hidden sm:inline"> Formula</span>
          </button>
          {imagesEnabled && (
            <>
              <input
                ref={fileRef}
                type="file"
                accept={ACCEPTED_IMAGE_TYPES.join(",")}
                hidden
                aria-label="Choose an image of a formula"
                onChange={(e) => {
                  addImage(e.target.files?.[0]);
                  e.target.value = ""; // the same file can be chosen again
                }}
              />
              <button
                type="button"
                onClick={() => fileRef.current?.click()}
                disabled={disabled}
                title="Read a formula from an image"
                aria-label="Add image"
                className="flex h-9 shrink-0 items-center gap-1.5 rounded-lg px-2.5 text-sm text-muted hover:bg-sunk hover:text-ink disabled:opacity-40"
              >
                <svg
                  viewBox="0 0 16 16"
                  width="16"
                  height="16"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.4"
                  aria-hidden="true"
                >
                  <rect x="1.5" y="2.5" width="13" height="11" rx="1.5" />
                  <circle cx="5.5" cy="6" r="1.1" />
                  <path d="M2 12.5l4-4 3 3 2-2 3 3" />
                </svg>
                <span className="hidden sm:inline">Image</span>
              </button>
            </>
          )}
          <span className="ml-auto hidden pr-2 text-xs text-muted md:inline" aria-hidden="true">
            Ctrl/⌘ + Enter to send
          </span>
          {busy ? (
            <button
              type="button"
              onClick={onStop}
              className="ml-auto h-9 shrink-0 rounded-lg border border-rule px-3 text-sm hover:bg-sunk md:ml-0"
            >
              Stop
            </button>
          ) : (
            <button
              type="button"
              onClick={submit}
              disabled={disabled || !text.trim()}
              className="ml-auto h-9 shrink-0 rounded-lg bg-accent px-4 text-sm font-medium text-accent-ink disabled:opacity-40 md:ml-0"
            >
              Send
            </button>
          )}
        </div>
      </div>
      <p className="mt-1.5 text-center text-xs text-muted">
        Results come from checked computation. Formulas written by the model alone are marked.
      </p>
    </div>
  );
}
