"use client";

import { useState } from "react";
import { ApiError } from "@/lib/api";
import type { PerturbationRequest } from "@/lib/chat-types";
import { buildPerturbationRequest, type PerturbationFormValues } from "@/lib/perturbation-form";
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

/** Asks for the numbers only the user can choose, then compares the solution with the numerical one. */
export function PerturbationForm({
  base,
  parameters,
}: {
  base: PerturbationRequest;
  parameters: string[];
}) {
  const runner = useMotionRunner();
  const [form, setForm] = useState<PerturbationFormValues>({
    parameters: {},
    epsilon: "",
    amplitude: "",
    tEnd: "",
  });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  if (!runner?.perturb) return null;
  const perturb = runner.perturb;
  const small = base.small_parameter;

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    const built = buildPerturbationRequest(base, parameters, form);
    if ("error" in built) return setError(built.error);
    setBusy(true);
    try {
      await perturb(built.request);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "The comparison could not be run.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <details className="mt-4 rounded border border-rule p-3">
      <summary className="cursor-pointer text-sm font-medium">
        Compare with the numerical solution{" "}
        <span className="font-normal text-muted">(you choose the numbers): show</span>
      </summary>
      <form onSubmit={submit} className="mt-3 space-y-4" aria-label="Comparison numbers">
        {parameters.length > 0 && (
          <fieldset>
            <legend className="text-sm font-medium">Parameters</legend>
            <div className="mt-1 grid grid-cols-2 gap-2 sm:grid-cols-4">
              {parameters.map((p) => (
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
          </fieldset>
        )}
        <fieldset>
          <legend className="text-sm font-medium">Size</legend>
          <div className="mt-1 grid grid-cols-2 gap-2 sm:grid-cols-4">
            <NumberInput
              label={small ? `${small} (the small parameter)` : "initial displacement"}
              value={form.epsilon}
              onChange={(v) => setForm((f) => ({ ...f, epsilon: v }))}
            />
            {small && (
              <NumberInput
                label="initial displacement A"
                value={form.amplitude}
                onChange={(v) => setForm((f) => ({ ...f, amplitude: v }))}
              />
            )}
            <NumberInput
              label="end time"
              value={form.tEnd}
              onChange={(v) => setForm((f) => ({ ...f, tEnd: v }))}
            />
          </div>
        </fieldset>
        <p className="text-xs text-muted">
          The solution starts at rest. It is compared at the value you give and at half of it; the
          error is a measurement on this span, not a bound.
        </p>
        {error && (
          <p role="alert" className="rounded bg-bad-soft px-3 py-2 text-sm text-bad">
            {error}
          </p>
        )}
        <button
          type="submit"
          disabled={busy}
          className="rounded bg-accent px-4 py-2 text-sm font-medium text-accent-ink disabled:opacity-50"
        >
          {busy ? "Comparing…" : "Compare"}
        </button>
      </form>
    </details>
  );
}
