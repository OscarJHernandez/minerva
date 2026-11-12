"use client";

import { useState } from "react";
import { ApiError } from "@/lib/api";
import type { MotionRequest } from "@/lib/chat-types";
import { buildMotionRequest, type MotionFormValues } from "@/lib/motion-form";
import { useMotionRunner } from "./motion-runner";

const field =
  "w-full min-w-0 rounded border border-rule bg-paper px-2 py-1 text-sm focus:border-accent";

function NumberInput({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <label className="block text-xs text-muted">
      {label}
      <input
        type="number"
        step="any"
        inputMode="decimal"
        className={field}
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
    </label>
  );
}

/** Asks for the numbers only the user can choose, then starts the verified numerical solution. */
export function MotionForm({
  base,
  parameters,
}: {
  base: Pick<MotionRequest, "lagrangian" | "coordinates" | "time" | "assumptions">;
  parameters: string[];
}) {
  const runner = useMotionRunner();
  const n = base.coordinates.length;
  const [form, setForm] = useState<MotionFormValues>({
    parameters: {},
    positions: Array<string>(n).fill(""),
    velocities: Array<string>(n).fill(""),
    tStart: "0",
    tEnd: "",
    sweep: null,
  });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  if (!runner) return null;

  const swept = form.sweep?.name ?? "";

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    const built = buildMotionRequest(base, parameters, form);
    if ("error" in built) return setError(built.error);
    setBusy(true);
    try {
      await runner?.run(built.request);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "The numerical solution could not be run.");
    } finally {
      setBusy(false);
    }
  }

  function setList(key: "positions" | "velocities", i: number, v: string) {
    setForm((f) => ({ ...f, [key]: f[key].map((x, k) => (k === i ? v : x)) }));
  }

  return (
    <details className="mt-4 rounded border border-rule p-3">
      <summary className="cursor-pointer text-sm font-medium">
        Solve numerically{" "}
        <span className="font-normal text-muted">(you choose the numbers): show</span>
      </summary>
      <form onSubmit={submit} className="mt-3 space-y-4" aria-label="Numerical solution">
        {parameters.length > 0 && (
          <fieldset>
            <legend className="text-sm font-medium">Parameters</legend>
            <div className="mt-1 grid grid-cols-2 gap-2 sm:grid-cols-4">
              {parameters
                .filter((p) => p !== swept)
                .map((p) => (
                  <NumberInput
                    key={p}
                    label={p}
                    value={form.parameters[p] ?? ""}
                    onChange={(v) =>
                      setForm((f) => ({ ...f, parameters: { ...f.parameters, [p]: v } }))
                    }
                  />
                ))}
            </div>
            <label className="mt-2 block text-xs text-muted">
              Sweep one parameter (2 to 6 values)
              <select
                className={`${field} mt-0.5 sm:w-auto`}
                value={swept}
                onChange={(e) =>
                  setForm((f) => ({
                    ...f,
                    sweep: e.target.value
                      ? { name: e.target.value, from: "", to: "", count: "3" }
                      : null,
                  }))
                }
              >
                <option value="">None, a single run</option>
                {parameters.map((p) => (
                  <option key={p} value={p}>
                    {p}
                  </option>
                ))}
              </select>
            </label>
            {form.sweep && (
              <div className="mt-2 grid grid-cols-3 gap-2 sm:max-w-md">
                <NumberInput
                  label={`${swept} from`}
                  value={form.sweep.from}
                  onChange={(v) =>
                    setForm((f) => ({ ...f, sweep: f.sweep && { ...f.sweep, from: v } }))
                  }
                />
                <NumberInput
                  label={`${swept} to`}
                  value={form.sweep.to}
                  onChange={(v) =>
                    setForm((f) => ({ ...f, sweep: f.sweep && { ...f.sweep, to: v } }))
                  }
                />
                <NumberInput
                  label="values"
                  value={form.sweep.count}
                  onChange={(v) =>
                    setForm((f) => ({ ...f, sweep: f.sweep && { ...f.sweep, count: v } }))
                  }
                />
              </div>
            )}
          </fieldset>
        )}

        <fieldset>
          <legend className="text-sm font-medium">Initial conditions</legend>
          <div className="mt-1 grid grid-cols-2 gap-2 sm:grid-cols-4">
            {base.coordinates.map((c, i) => (
              <div key={c} className="contents">
                <NumberInput
                  label={`${c}(start)`}
                  value={form.positions[i] ?? ""}
                  onChange={(v) => setList("positions", i, v)}
                />
                <NumberInput
                  label={`d${c}/dt (start)`}
                  value={form.velocities[i] ?? ""}
                  onChange={(v) => setList("velocities", i, v)}
                />
              </div>
            ))}
          </div>
        </fieldset>

        <fieldset>
          <legend className="text-sm font-medium">Time span</legend>
          <div className="mt-1 grid grid-cols-2 gap-2 sm:max-w-xs">
            <NumberInput
              label="from"
              value={form.tStart}
              onChange={(v) => setForm((f) => ({ ...f, tStart: v }))}
            />
            <NumberInput
              label="to"
              value={form.tEnd}
              onChange={(v) => setForm((f) => ({ ...f, tEnd: v }))}
            />
          </div>
        </fieldset>

        {error && (
          <p role="alert" className="rounded bg-bad-soft px-3 py-2 text-sm text-bad">
            {error}
          </p>
        )}
        <div className="flex flex-wrap items-center gap-3">
          <button
            type="submit"
            disabled={busy}
            className="whitespace-nowrap rounded bg-accent px-3 py-1.5 text-sm font-medium text-accent-ink disabled:opacity-50"
          >
            {busy ? "Solving and verifying…" : "Solve and verify"}
          </button>
          <span className="text-xs text-muted">
            Each run is shown only if it passes Minerva&apos;s checks.
          </span>
        </div>
      </form>
    </details>
  );
}
