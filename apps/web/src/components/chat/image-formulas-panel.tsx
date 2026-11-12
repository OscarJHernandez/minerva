"use client";

import { useEffect, useState } from "react";
import { Math } from "@minerva/ui/math";
import { ApiError } from "@/lib/api";
import { readFormulaImage } from "@/lib/chat-api";
import type { ExtractedFormula } from "@/lib/chat-types";
import { ImageError, prepareImage } from "@/lib/image-prep";
import { MathField } from "./math-field";

interface Item {
  original: ExtractedFormula;
  latex: string;
  edited: boolean;
  selected: boolean;
  editing: boolean;
}

type Phase =
  { kind: "reading" } | { kind: "error"; message: string } | { kind: "found"; items: Item[] };

/**
 * Reads the formulas in a picture and asks the user to confirm them before they go into the message.
 * The model only transcribed; "Minerva reads this as" is the strict parser's own account of it.
 */
export function ImageFormulasPanel({
  file,
  onInsert,
  onClose,
}: {
  file: File;
  onInsert: (latex: string[]) => void;
  onClose: () => void;
}) {
  const [phase, setPhase] = useState<Phase>({ kind: "reading" });
  const [preview, setPreview] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    let url: string | null = null;
    (async () => {
      try {
        const prepared = await prepareImage(file);
        url = prepared.previewUrl;
        if (controller.signal.aborted) return;
        setPreview(url);
        const formulas = await readFormulaImage(prepared.base64, controller.signal);
        setPhase({
          kind: "found",
          items: formulas.map((original) => ({
            original,
            latex: original.latex,
            edited: false,
            selected: original.status === "ok",
            editing: false,
          })),
        });
      } catch (e) {
        if (controller.signal.aborted) return;
        setPhase({ kind: "error", message: describe(e) });
      }
    })();
    return () => {
      controller.abort();
      if (url) URL.revokeObjectURL(url);
    };
  }, [file]);

  function update(index: number, patch: Partial<Item>) {
    setPhase((p) =>
      p.kind === "found"
        ? { kind: "found", items: p.items.map((it, i) => (i === index ? { ...it, ...patch } : it)) }
        : p,
    );
  }

  const items = phase.kind === "found" ? phase.items : [];
  const chosen = items.filter((it) => it.selected && it.latex.trim());

  return (
    <div
      className="mb-2 rounded-lg border border-rule bg-raised p-3 shadow-sm"
      role="dialog"
      aria-label="Formulas from image"
    >
      <div className="flex flex-col gap-3 sm:flex-row">
        {preview && (
          // eslint-disable-next-line @next/next/no-img-element -- a local blob URL, not a remote image
          <img
            src={preview}
            alt="The image you added"
            className="h-16 w-full shrink-0 rounded border border-rule bg-sunk object-contain sm:h-20 sm:w-20"
          />
        )}
        <div className="min-w-0 flex-1">
          {phase.kind === "reading" && (
            <p className="text-sm text-muted" aria-live="polite">
              Reading the image…
            </p>
          )}
          {phase.kind === "error" && (
            <p role="alert" className="rounded bg-bad-soft px-3 py-2 text-sm text-bad">
              {phase.message}
            </p>
          )}
          {phase.kind === "found" && items.length === 0 && (
            <p className="text-sm">
              No formula was found in this image. Try a closer, sharper picture of just the formula.
            </p>
          )}
          {items.length > 0 && (
            <>
              <p className="text-sm font-medium">Is this what the image says?</p>
              <p className="mb-2 text-xs text-muted">
                A model transcribed the image; it can be wrong. Check each formula against the image
                before you use it.
              </p>
              <ul className="space-y-2">
                {items.map((item, i) => (
                  <li key={i} className="rounded border border-rule p-2">
                    <div className="flex items-start gap-2">
                      <input
                        type="checkbox"
                        checked={item.selected}
                        onChange={(e) => update(i, { selected: e.target.checked })}
                        aria-label={`Use formula ${i + 1}`}
                        className="mt-1.5"
                      />
                      <div className="min-w-0 flex-1 overflow-x-auto">
                        <Math latex={item.latex} display />
                      </div>
                      <button
                        type="button"
                        onClick={() => update(i, { editing: !item.editing })}
                        className="shrink-0 text-xs text-accent hover:underline"
                      >
                        {item.editing ? "Done" : "Edit"}
                      </button>
                    </div>
                    {item.editing && (
                      // Its own full-width row: the editor's toolbar needs room beside the formula.
                      <div className="mt-2">
                        <MathField
                          label={`Edit formula ${i + 1}`}
                          initial={item.latex}
                          onChange={(latex) => update(i, { latex, edited: true })}
                          onEnter={() => update(i, { editing: false })}
                          onEscape={() => update(i, { editing: false })}
                        />
                      </div>
                    )}
                    <Reading item={item} />
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
      </div>
      <div className="mt-3 flex justify-end gap-2">
        <button
          type="button"
          onClick={onClose}
          className="rounded border border-rule px-3 py-1.5 text-sm hover:bg-sunk"
        >
          {items.length > 0 ? "Discard" : "Close"}
        </button>
        {items.length > 0 && (
          <button
            type="button"
            disabled={chosen.length === 0}
            onClick={() => onInsert(chosen.map((it) => it.latex.trim()))}
            className="rounded bg-accent px-3 py-1.5 text-sm font-medium text-accent-ink disabled:opacity-50"
          >
            Use selected ({chosen.length})
          </button>
        )}
      </div>
    </div>
  );
}

function Reading({ item }: { item: Item }) {
  if (item.edited) {
    return (
      <p className="mt-1 text-xs text-muted">
        Edited by you. Minerva has not re-read this version.
      </p>
    );
  }
  const { status, read_as, message } = item.original;
  if (status === "ok" && read_as) {
    return (
      <div className="mt-1 text-xs text-muted">
        <span>Minerva reads this as </span>
        <Math latex={read_as} />
      </div>
    );
  }
  return (
    <p className="mt-1 text-xs text-warn">
      Minerva cannot read this formula{message ? `: ${message.replace(/[.\s]+$/, "")}.` : "."} You
      can edit it, or use it as it is and say what it means in your message.
    </p>
  );
}

function describe(e: unknown): string {
  if (e instanceof ImageError) return e.message;
  if (e instanceof ApiError) {
    if (e.status === 401) return "Your session has ended. Sign in again.";
    return e.message;
  }
  return "Could not read the image. Try again.";
}
