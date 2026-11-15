"use client";

import { useId } from "react";
import { FIELDS_FOR, type ApproximationForm } from "@/lib/approximation-form";
import { METHOD_LABEL, type ApproximationMethod } from "@/lib/approximation-types";
import { FIELD } from "./variation-fields";

type Key = keyof ApproximationForm;

/** Labels and hints. A hint says what to type; none of them is a value that gets submitted. */
const TEXT: Partial<Record<Key, { label: string; hint: string }>> = {
  point: { label: "Point", hint: "a number, or oo / -oo" },
  small: { label: "Small (or large) parameter", hint: "e.g. eps, or v/c" },
  lower: { label: "Lower end", hint: "a number" },
  upper: { label: "Upper end", hint: "a number" },
  period: { label: "Period", hint: "e.g. 2*pi" },
  kind: { label: "Kind", hint: "pchip, cubic, linear; sigma for Fourier" },
  variables: { label: "Variables", hint: "x, y" },
  points: { label: "Their points", hint: "0, 0" },
};
const NUMBER: Partial<Record<Key, string>> = {
  order: "Order",
  degree: "Degree (numerator degree for Padé)",
  denominatorDegree: "Denominator degree",
  modes: "Fourier modes",
  segments: "Segments",
  maxTerms: "Most terms allowed",
};

/**
 * The numbers of an approximation. They are the user's: nothing is pre-filled except what the user
 * already wrote in their message, and the fields shown depend on the method.
 */
export function ApproximationFields({
  form,
  onChange,
  disabled,
}: {
  form: ApproximationForm;
  onChange: (next: ApproximationForm) => void;
  disabled?: boolean;
}) {
  const id = useId();
  const shown = new Set(FIELDS_FOR[form.method]);
  const set = <K extends Key>(key: K, value: ApproximationForm[K]) =>
    onChange({ ...form, [key]: value });

  return (
    <div className="grid gap-3 sm:grid-cols-2">
      <label className="block text-sm sm:col-span-2" htmlFor={`${id}-method`}>
        Method
        <select
          id={`${id}-method`}
          className={`${FIELD} mt-1 w-full`}
          value={form.method}
          disabled={disabled}
          onChange={(e) => set("method", e.target.value as ApproximationMethod)}
        >
          {(Object.keys(METHOD_LABEL) as ApproximationMethod[]).map((m) => (
            <option key={m} value={m}>
              {METHOD_LABEL[m]}
            </option>
          ))}
        </select>
      </label>
      {(Object.keys(TEXT) as Key[])
        .filter((k) => shown.has(k))
        .map((k) => (
          <label key={k} className="block text-sm" htmlFor={`${id}-${k}`}>
            {TEXT[k]?.label}
            <input
              id={`${id}-${k}`}
              className={`${FIELD} mt-1 w-full`}
              value={form[k]}
              disabled={disabled}
              placeholder={TEXT[k]?.hint}
              onChange={(e) => set(k, e.target.value as ApproximationForm[typeof k])}
              autoComplete="off"
              spellCheck={false}
            />
          </label>
        ))}
      {(Object.keys(NUMBER) as Key[])
        .filter((k) => shown.has(k))
        .map((k) => (
          <label key={k} className="block text-sm" htmlFor={`${id}-${k}`}>
            {NUMBER[k]}
            <input
              id={`${id}-${k}`}
              className={`${FIELD} mt-1 w-full`}
              inputMode="numeric"
              value={form[k]}
              disabled={disabled}
              onChange={(e) => set(k, e.target.value as ApproximationForm[typeof k])}
              autoComplete="off"
            />
          </label>
        ))}
      {shown.has("direction") && (
        <label className="block text-sm" htmlFor={`${id}-direction`}>
          Side
          <select
            id={`${id}-direction`}
            className={`${FIELD} mt-1 w-full`}
            value={form.direction}
            disabled={disabled}
            onChange={(e) => set("direction", e.target.value as ApproximationForm["direction"])}
          >
            <option value="">both sides</option>
            <option value="+">from above (+)</option>
            <option value="-">from below (−)</option>
          </select>
        </label>
      )}
      {shown.has("tolerance") && (
        <fieldset className="grid grid-cols-2 gap-3 sm:col-span-2">
          <label className="block text-sm" htmlFor={`${id}-tolerance`}>
            Accuracy target
            <input
              id={`${id}-tolerance`}
              className={`${FIELD} mt-1 w-full`}
              value={form.tolerance}
              disabled={disabled}
              placeholder="e.g. 1e-6"
              onChange={(e) => set("tolerance", e.target.value)}
              autoComplete="off"
              spellCheck={false}
            />
          </label>
          <label className="block text-sm" htmlFor={`${id}-toleranceKind`}>
            Kind of error
            <select
              id={`${id}-toleranceKind`}
              className={`${FIELD} mt-1 w-full`}
              value={form.toleranceKind}
              disabled={disabled}
              onChange={(e) =>
                set("toleranceKind", e.target.value as ApproximationForm["toleranceKind"])
              }
            >
              <option value="">choose…</option>
              <option value="absolute">absolute</option>
              <option value="relative">relative</option>
            </select>
          </label>
        </fieldset>
      )}
    </div>
  );
}
