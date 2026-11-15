"use client";

import Link from "next/link";
import { useState } from "react";
import { ApiError, api } from "@/lib/api";
import {
  blankForm,
  buildApproximationNumbers,
  restoreApproximationForm,
  type ApproximationForm,
} from "@/lib/approximation-form";
import type {
  ApproximationNumbers,
  ApproximationReport,
  ApproximationRequest,
} from "@/lib/approximation-types";
import { ApproximationFields } from "./approximation-fields";
import { ApproximationPanel } from "./approximation-panel";
import { MathInput } from "./math-input";

const NAME = /^[A-Za-z][A-Za-z0-9_]{0,31}$/;
const FIELD =
  "w-full min-w-0 rounded border border-rule bg-raised px-2 py-1.5 text-sm text-ink disabled:opacity-50";

/** Examples fill the form when clicked; nothing is submitted or defaulted without the user. */
const EXAMPLES: { label: string; latex: string; numbers: ApproximationNumbers }[] = [
  {
    label: "Padé [1/1] of the exponential",
    latex: "\\exp(x)",
    numbers: { method: "pade", point: "0", degree: 1, denominator_degree: 1 },
  },
  {
    label: "Large x behaviour",
    latex: "\\frac{1}{x+1}",
    numbers: { method: "asymptotic", point: "oo", order: 3 },
  },
  {
    label: "Compare on an interval",
    latex: "\\log(1+x)",
    numbers: { method: "compare", lower: "0", upper: "2", degree: 5 },
  },
  {
    label: "Rational approximations of π",
    latex: "\\pi",
    numbers: { method: "continued_fraction", order: 6 },
  },
];

function names(raw: string): string[] {
  return raw
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s !== "");
}

/** Validates the page; returns the request or a message. Exported for tests. */
export function buildApproximationRequest(input: {
  expression: string;
  variable: string;
  positive: string;
  real: string;
  form: ApproximationForm;
}): ApproximationRequest | string {
  if (!input.expression.trim()) return "Enter the expression.";
  const variable = input.variable.trim() || "x";
  if (!NAME.test(variable)) return "The variable is a name made of letters and digits.";
  const assumptions: Record<string, string[]> = {};
  for (const [list, kind] of [
    [names(input.positive), "positive"],
    [names(input.real), "real"],
  ] as const) {
    for (const name of list) {
      if (!NAME.test(name)) return `“${name}” is not a symbol name.`;
      assumptions[name] = [kind];
    }
  }
  const built = buildApproximationNumbers(input.form);
  if (!built.ok) return built.error;
  return {
    input: input.expression,
    format: "latex",
    variable,
    assumptions,
    ...built.numbers,
    save: false,
  };
}

/**
 * Approximations on their own page: an expression, a method (or "choose automatically") and the
 * numbers that say where the approximation is wanted. Every number is yours; the engine shows a
 * result only if a check that does not use the route that built it agrees.
 */
export function ApproximationWorkbench() {
  const [expression, setExpression] = useState("");
  const [variable, setVariable] = useState("x");
  const [positive, setPositive] = useState("");
  const [real, setReal] = useState("");
  const [form, setForm] = useState<ApproximationForm>(blankForm());
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<ApproximationReport | null>(null);

  async function run() {
    const request = buildApproximationRequest({ expression, variable, positive, real, form });
    if (typeof request === "string") {
      setError(request);
      return;
    }
    setError(null);
    setBusy(true);
    try {
      setReport(await api.approximation(request));
    } catch (e) {
      setReport(null);
      setError(e instanceof ApiError ? e.message : "The approximation failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-5 pb-16 pt-8">
      <div>
        <p className="text-xs text-muted">
          <Link href="/workbench" className="text-accent hover:underline">
            Workbench
          </Link>{" "}
          / Approximations
        </p>
        <h1 className="font-serif text-2xl">Approximations</h1>
        <p className="mt-1 max-w-2xl text-sm text-muted">
          Find a simpler expression that represents a function where you need it: near a point, for
          large x, in a small parameter, on an interval, for a periodic function, for a large sum.
          Minerva shows an approximation only if a check that does not use the route that built it
          agrees. Errors are measured on sampled points; they are not bounds.
        </p>
      </div>

      <div className="flex flex-wrap gap-2" aria-label="Examples">
        {EXAMPLES.map((ex) => (
          <button
            key={ex.label}
            type="button"
            className="rounded-full border border-rule px-3 py-1 text-xs hover:bg-sunk"
            onClick={() => {
              setExpression(ex.latex);
              setForm(restoreApproximationForm(ex.numbers));
            }}
          >
            {ex.label}
          </button>
        ))}
      </div>

      <MathInput value={expression} onChange={setExpression} onSubmit={run} />
      <p className="-mt-3 text-xs text-muted">
        For a sum, the summand; for a number, the number. Use \exp(x) for the exponential (a bare e
        is a symbol).
      </p>

      <div className="grid gap-3 sm:grid-cols-3">
        <label className="block text-sm">
          Variable
          <input className={FIELD} value={variable} onChange={(e) => setVariable(e.target.value)} />
        </label>
        <label className="block text-sm">
          Symbols known positive
          <input
            className={FIELD}
            value={positive}
            onChange={(e) => setPositive(e.target.value)}
            placeholder="c, m"
          />
        </label>
        <label className="block text-sm">
          Symbols known real
          <input
            className={FIELD}
            value={real}
            onChange={(e) => setReal(e.target.value)}
            placeholder="v"
          />
        </label>
      </div>

      <ApproximationFields form={form} onChange={setForm} disabled={busy} />

      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={run}
          disabled={busy}
          className="rounded bg-accent px-4 py-2 text-sm font-medium text-accent-ink disabled:opacity-50"
        >
          {busy ? "Approximating and checking…" : "Approximate"}
        </button>
        {busy && <span className="text-xs text-muted">This can take several seconds.</span>}
      </div>
      {error && (
        <p role="alert" className="rounded bg-bad-soft px-3 py-2 text-sm text-bad">
          {error}
        </p>
      )}
      {report && <ApproximationPanel report={report} />}
    </div>
  );
}
