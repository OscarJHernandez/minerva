"use client";

import { useState } from "react";
import { ASSUMPTIONS, type Assumption, type Assumptions } from "@/lib/types";

interface Props {
  value: Assumptions;
  onChange: (next: Assumptions) => void;
}

const NAME_RE = /^[A-Za-z][A-Za-z0-9_]{0,31}$/;

/** Declares properties of symbols (e.g. x positive). Results are only valid under them. */
export function AssumptionsEditor({ value, onChange }: Props) {
  const [name, setName] = useState("");
  const symbols = Object.keys(value);

  function addSymbol() {
    const trimmed = name.trim();
    if (!NAME_RE.test(trimmed) || value[trimmed]) return;
    onChange({ ...value, [trimmed]: ["real"] });
    setName("");
  }

  function toggle(symbol: string, a: Assumption) {
    const current = value[symbol] ?? [];
    const next = current.includes(a) ? current.filter((x) => x !== a) : [...current, a];
    onChange({ ...value, [symbol]: next });
  }

  function remove(symbol: string) {
    const next = { ...value };
    delete next[symbol];
    onChange(next);
  }

  return (
    <section aria-labelledby="assumptions-heading">
      <h2 id="assumptions-heading" className="text-sm font-medium">
        Assumptions
      </h2>
      <p className="mb-2 text-xs text-muted">
        Symbols are complex by default. Declare properties to unlock identities that need them.
      </p>
      <ul className="space-y-2">
        {symbols.map((s) => (
          <li key={s} className="flex items-start gap-2 text-sm">
            <code className="mt-0.5 w-12 shrink-0 rounded bg-sunk px-1.5 py-0.5 text-center font-mono">
              {s}
            </code>
            <div className="flex min-w-0 flex-1 flex-wrap gap-1.5">
              {ASSUMPTIONS.map((a) => {
                const on = value[s]?.includes(a) ?? false;
                return (
                  <button
                    key={a}
                    type="button"
                    aria-pressed={on}
                    onClick={() => toggle(s, a)}
                    className={`rounded-full border px-2 py-0.5 text-xs ${
                      on
                        ? "border-accent bg-accent-soft text-accent"
                        : "border-rule text-muted hover:bg-sunk"
                    }`}
                  >
                    {a}
                  </button>
                );
              })}
            </div>
            <button
              type="button"
              onClick={() => remove(s)}
              aria-label={`Remove assumptions for ${s}`}
              className="mt-0.5 shrink-0 text-xs text-muted hover:text-bad"
            >
              remove
            </button>
          </li>
        ))}
      </ul>
      <form
        className="mt-2 flex gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          addSymbol();
        }}
      >
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="symbol, e.g. x"
          aria-label="Symbol name"
          className="w-36 rounded border border-rule bg-raised px-2 py-1 text-sm"
        />
        <button
          type="submit"
          className="rounded border border-rule px-2.5 py-1 text-sm hover:bg-sunk"
        >
          Add
        </button>
      </form>
    </section>
  );
}
