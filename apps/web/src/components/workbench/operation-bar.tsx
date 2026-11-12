"use client";

import { useRef } from "react";
import { MAX_DERIVATIVE_ORDER, MAX_SERIES_ORDER } from "@/lib/types";
import type { Settings, StepOperation } from "@/lib/derivation";

interface Props {
  /** The operation whose settings are open, if it needs any. */
  selected: StepOperation | null;
  onSelect: (operation: StepOperation | null) => void;
  settings: Settings;
  onSettings: (settings: Settings) => void;
  onRun: (operation: StepOperation) => void;
  onFind: () => void;
  /** No expression yet. */
  empty: boolean;
  busy: string | null;
}

const OPERATIONS: { id: StepOperation | "identities"; label: string; needsSettings: boolean }[] = [
  { id: "simplify", label: "Simplify", needsSettings: false },
  { id: "differentiate", label: "Differentiate", needsSettings: true },
  { id: "integrate", label: "Integrate", needsSettings: true },
  { id: "series", label: "Series", needsSettings: true },
  { id: "identities", label: "Identities", needsSettings: false },
];

const FIELD =
  "rounded border border-rule bg-raised px-2 py-1.5 text-sm text-ink disabled:opacity-50";

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex items-center gap-1.5 text-xs text-muted">
      {label}
      {children}
    </label>
  );
}

/** Operation buttons plus the settings of the one that needs them: no parameters compete for space. */
export function OperationBar({
  selected,
  onSelect,
  settings,
  onSettings,
  onRun,
  onFind,
  empty,
  busy,
}: Props) {
  const bar = useRef<HTMLDivElement | null>(null);
  const set = (patch: Partial<Settings>) => onSettings({ ...settings, ...patch });
  const disabled = empty || busy !== null;

  function arrows(e: React.KeyboardEvent) {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    const buttons = Array.from(bar.current?.querySelectorAll<HTMLButtonElement>("button") ?? []);
    const at = buttons.indexOf(document.activeElement as HTMLButtonElement);
    if (at < 0) return;
    e.preventDefault();
    const step = e.key === "ArrowRight" ? 1 : -1;
    buttons[(at + step + buttons.length) % buttons.length]?.focus();
  }

  return (
    <div>
      <div
        ref={bar}
        role="toolbar"
        aria-label="Operations"
        onKeyDown={arrows}
        className="flex flex-wrap gap-1.5"
      >
        {OPERATIONS.map((op) => {
          const open = selected === op.id;
          return (
            <button
              key={op.id}
              type="button"
              disabled={disabled}
              aria-pressed={op.needsSettings ? open : undefined}
              aria-expanded={op.needsSettings ? open : undefined}
              onClick={() => {
                if (op.id === "identities") onFind();
                else if (op.needsSettings) onSelect(open ? null : (op.id as StepOperation));
                else onRun(op.id as StepOperation);
              }}
              className={`rounded border px-3.5 py-1.5 text-sm font-medium disabled:opacity-50 ${
                open
                  ? "border-accent bg-accent-soft text-accent"
                  : op.id === "simplify"
                    ? "border-accent bg-accent text-accent-ink"
                    : "border-rule text-ink hover:bg-sunk"
              }`}
            >
              {busy === op.id ? "Working…" : op.label}
            </button>
          );
        })}
      </div>

      {selected && selected !== "simplify" && (
        <form
          aria-label={`${selected} settings`}
          className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2 rounded-lg border border-rule bg-sunk px-3 py-2.5"
          onSubmit={(e) => {
            e.preventDefault();
            onRun(selected);
          }}
        >
          <Field label="by">
            <input
              value={settings.variable}
              onChange={(e) => set({ variable: e.target.value.trim() })}
              aria-label="Variable"
              maxLength={32}
              className={`${FIELD} w-16 font-mono`}
            />
          </Field>
          {selected === "differentiate" && (
            <Field label="order">
              <input
                type="number"
                min={1}
                max={MAX_DERIVATIVE_ORDER}
                value={settings.order}
                onChange={(e) =>
                  set({
                    order: Math.min(MAX_DERIVATIVE_ORDER, Math.max(1, Number(e.target.value) || 1)),
                  })
                }
                aria-label="Order of the derivative"
                className={`${FIELD} w-16`}
              />
            </Field>
          )}
          {selected === "integrate" && (
            <>
              <Field label="from">
                <input
                  value={settings.lower}
                  onChange={(e) => set({ lower: e.target.value })}
                  aria-label="Lower bound (optional)"
                  placeholder="optional"
                  maxLength={200}
                  className={`${FIELD} w-20 font-mono`}
                />
              </Field>
              <Field label="to">
                <input
                  value={settings.upper}
                  onChange={(e) => set({ upper: e.target.value })}
                  aria-label="Upper bound (optional)"
                  placeholder="optional"
                  maxLength={200}
                  className={`${FIELD} w-20 font-mono`}
                />
              </Field>
            </>
          )}
          {selected === "series" && (
            <>
              <Field label="about">
                <input
                  value={settings.point}
                  onChange={(e) => set({ point: e.target.value.trim() })}
                  aria-label="Point to expand about"
                  maxLength={64}
                  className={`${FIELD} w-16 font-mono`}
                />
              </Field>
              <Field label="to order">
                <input
                  type="number"
                  min={1}
                  max={MAX_SERIES_ORDER}
                  value={settings.seriesOrder}
                  onChange={(e) =>
                    set({
                      seriesOrder: Math.min(
                        MAX_SERIES_ORDER,
                        Math.max(1, Number(e.target.value) || 1),
                      ),
                    })
                  }
                  aria-label="Order of the series"
                  className={`${FIELD} w-16`}
                />
              </Field>
            </>
          )}
          <button
            type="submit"
            disabled={disabled}
            className="ml-auto rounded bg-accent px-4 py-1.5 text-sm font-medium text-accent-ink disabled:opacity-50"
          >
            {busy === selected ? "Working…" : "Run"}
          </button>
        </form>
      )}
    </div>
  );
}
