"use client";

import { useId, useState } from "react";
import { ApiError } from "@/lib/api";
import {
  blankNumericForm,
  buildNumericRequest,
  buildSymbolicRequest,
  prefillFromConditions,
  stateFields,
  stateKey,
  type NumericForm,
} from "@/lib/ode-form";
import type { NumericMethod, OdeInput, OdeInterpretation, OdeNumericInput } from "@/lib/ode-types";

const FIELD =
  "w-full min-w-0 rounded border border-rule bg-paper px-2 py-1 text-sm focus:border-accent";

function Box({
  label,
  value,
  onChange,
  hint,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  hint?: string;
}) {
  return (
    <label className="block text-xs text-muted">
      {label}
      <input
        type="text"
        inputMode="decimal"
        autoComplete="off"
        spellCheck={false}
        placeholder={hint}
        className={FIELD}
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
    </label>
  );
}

const METHODS: { value: NumericMethod; label: string }[] = [
  { value: "auto", label: "Automatic (chosen from measured behaviour)" },
  { value: "DOP853", label: "DOP853 (explicit, high order)" },
  { value: "RK45", label: "RK45 (explicit)" },
  { value: "Radau", label: "Radau (implicit, for stiff problems)" },
  { value: "BDF", label: "BDF (implicit, for stiff problems)" },
  { value: "LSODA", label: "LSODA (switches by itself)" },
];

/** The shape of an error shown to the user: the server's own message when there is one. */
function message(e: unknown, fallback: string): string {
  return e instanceof ApiError ? e.message : fallback;
}

/**
 * Asks for every number only the user can choose: where the initial values hold, each initial value,
 * each parameter, the end of the interval. Nothing is filled in on the user's behalf except what they
 * typed with the equation.
 */
export function OdeNumericForm({
  base,
  meaning,
  onRun,
}: {
  base: OdeInput;
  meaning: OdeInterpretation;
  onRun: (request: OdeNumericInput) => Promise<void>;
}) {
  const id = useId();
  const [form, setForm] = useState<NumericForm>(() => ({
    ...blankNumericForm(meaning),
    ...prefillFromConditions(base.conditions, meaning),
  }));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fields = stateFields(meaning);
  const point = meaning.independent;

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    const built = buildNumericRequest(base, meaning, form);
    if (!built.ok) return setError(built.error);
    setBusy(true);
    try {
      await onRun(built.request);
    } catch (err) {
      setError(message(err, "The numerical solution could not be run."));
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="space-y-4" aria-label="Numerical solution">
      <fieldset>
        <legend className="text-sm font-medium">Initial values</legend>
        <div className="mt-1 grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Box
            label={`at ${point} =`}
            value={form.start}
            onChange={(v) => setForm((f) => ({ ...f, start: v }))}
            hint="0"
          />
          {fields.map((f) => (
            <Box
              key={stateKey(f)}
              label={`${f.key}(${point}₀)`}
              value={form.initial[stateKey(f)] ?? ""}
              onChange={(v) =>
                setForm((s) => ({ ...s, initial: { ...s.initial, [stateKey(f)]: v } }))
              }
            />
          ))}
        </div>
      </fieldset>
      {meaning.parameters.length > 0 && (
        <fieldset>
          <legend className="text-sm font-medium">Parameters</legend>
          <div className="mt-1 grid grid-cols-2 gap-2 sm:grid-cols-4">
            {meaning.parameters.map((p) => (
              <Box
                key={p}
                label={p}
                value={form.parameters[p] ?? ""}
                onChange={(v) =>
                  setForm((s) => ({ ...s, parameters: { ...s.parameters, [p]: v } }))
                }
              />
            ))}
          </div>
        </fieldset>
      )}
      <fieldset>
        <legend className="text-sm font-medium">Interval</legend>
        <div className="mt-1 grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Box
            label={`${point} from ${form.start === "" ? "the start" : form.start} to`}
            value={form.tEnd}
            onChange={(v) => setForm((s) => ({ ...s, tEnd: v }))}
          />
        </div>
      </fieldset>
      <details>
        <summary className="cursor-pointer text-sm text-muted">Method and tolerances</summary>
        <div className="mt-2 grid gap-2 sm:grid-cols-3">
          <label className="block text-xs text-muted sm:col-span-1" htmlFor={`${id}-method`}>
            Integrator
            <select
              id={`${id}-method`}
              className={FIELD}
              value={form.method}
              onChange={(e) => setForm((s) => ({ ...s, method: e.target.value as NumericMethod }))}
            >
              {METHODS.map((m) => (
                <option key={m.value} value={m.value}>
                  {m.label}
                </option>
              ))}
            </select>
          </label>
          <Box
            label="Relative tolerance"
            value={form.rtol}
            onChange={(v) => setForm((s) => ({ ...s, rtol: v }))}
          />
          <Box
            label="Absolute tolerance"
            value={form.atol}
            onChange={(v) => setForm((s) => ({ ...s, atol: v }))}
          />
        </div>
      </details>
      {error && (
        <p role="alert" className="text-sm text-bad">
          {error}
        </p>
      )}
      <button
        type="submit"
        disabled={busy}
        className="rounded bg-accent px-4 py-2 text-sm font-medium text-accent-ink disabled:opacity-60"
      >
        {busy ? "Solving and checking…" : "Solve and check"}
      </button>
    </form>
  );
}

/** Optionally gives initial values, then asks for a closed form. Leave them empty for the general solution. */
export function OdeSymbolicForm({
  base,
  meaning,
  onRun,
}: {
  base: OdeInput;
  meaning: OdeInterpretation;
  onRun: (request: OdeInput) => Promise<void>;
}) {
  const [values, setValues] = useState(() => prefillFromConditions(base.conditions, meaning));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fields = stateFields(meaning);
  const point = meaning.independent;

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    const built = buildSymbolicRequest(base, meaning, values.start, values.initial);
    if (!built.ok) return setError(built.error);
    setBusy(true);
    try {
      await onRun(built.request);
    } catch (err) {
      setError(message(err, "The solution could not be run."));
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="space-y-3" aria-label="Closed-form solution">
      <fieldset>
        <legend className="text-sm font-medium">
          Initial values <span className="font-normal text-muted">(optional)</span>
        </legend>
        <div className="mt-1 grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Box
            label={`at ${point} =`}
            value={values.start}
            onChange={(v) => setValues((s) => ({ ...s, start: v }))}
            hint="0"
          />
          {fields.map((f) => (
            <Box
              key={stateKey(f)}
              label={`${f.key}(${point}₀)`}
              value={values.initial[stateKey(f)] ?? ""}
              onChange={(v) =>
                setValues((s) => ({ ...s, initial: { ...s.initial, [stateKey(f)]: v } }))
              }
            />
          ))}
        </div>
      </fieldset>
      {error && (
        <p role="alert" className="text-sm text-bad">
          {error}
        </p>
      )}
      <button
        type="submit"
        disabled={busy}
        className="rounded bg-accent px-4 py-2 text-sm font-medium text-accent-ink disabled:opacity-60"
      >
        {busy ? "Solving and checking…" : "Find a closed form"}
      </button>
    </form>
  );
}
