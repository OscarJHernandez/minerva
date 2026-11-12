"use client";

import { useCallback, useRef, useState } from "react";
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

export function Composer({ disabled, busy, onSend, onStop, imagesEnabled = false }: Props) {
  const [text, setText] = useState("");
  const [formula, setFormula] = useState(false);
  const [image, setImage] = useState<File | null>(null);
  const [imageError, setImageError] = useState<string | null>(null);
  const areaRef = useRef<HTMLTextAreaElement | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);

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
      <div className="flex items-end gap-2 rounded-2xl border border-rule bg-raised p-2 focus-within:outline focus-within:outline-2 focus-within:outline-accent">
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
        <textarea
          ref={areaRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onPaste={(e) => {
            const file = imagesEnabled
              ? Array.from(e.clipboardData.files).find((f) => f.type.startsWith("image/"))
              : undefined;
            if (!file) return; // plain text pastes as usual
            e.preventDefault();
            addImage(file);
          }}
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
        Results come from checked computation. Formulas written by the model alone are marked.
      </p>
    </div>
  );
}
