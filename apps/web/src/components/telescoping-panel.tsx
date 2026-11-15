"use client";

import Link from "next/link";
import { useState } from "react";
import { ApiError, api } from "@/lib/api";
import type { TelescopingRequest, TelescopingResult } from "@/lib/telescoping-types";
import { TelescopingCard } from "./chat/telescoping-card";
import { MathInput } from "./math-input";

const NAME = /^[A-Za-z][A-Za-z0-9_]{0,31}$/;
const FIELD =
  "w-full min-w-0 rounded border border-rule bg-raised px-2 py-1.5 text-sm text-ink disabled:opacity-50";

export type Sign = "positive" | "negative" | "real" | "none";

/** Validates the form; returns the request or a message. Exported for tests. */
export function buildTelescopingRequest(form: {
  integrand: string;
  variable: string;
  parameter: string;
  lower: string;
  upper: string;
  parameterSign: Sign;
}): TelescopingRequest | string {
  if (!form.integrand.trim()) return "Enter the integrand.";
  const variable = form.variable.trim();
  const parameter = form.parameter.trim();
  if (!NAME.test(variable) || !NAME.test(parameter))
    return "The variable and the parameter are names made of letters and digits.";
  if (variable === parameter) return "The variable and the parameter must differ.";
  const lower = form.lower.trim();
  const upper = form.upper.trim();
  if ((lower === "") !== (upper === "")) return "Give both bounds, or neither.";
  const assumptions: Record<string, string[]> = { [variable]: ["real"] };
  if (form.parameterSign !== "none") {
    assumptions[parameter] = form.parameterSign === "real" ? ["real"] : [form.parameterSign];
  }
  return {
    input: form.integrand,
    format: "latex",
    variable,
    parameter,
    lower: lower || null,
    upper: upper || null,
    assumptions,
  };
}

/**
 * Creative telescoping: give an integrand f(x, a), the parameter and the bounds. The signs you declare are
 * the only assumptions used; Minerva never assumes a parameter is positive.
 */
export function TelescopingPanel() {
  const [integrand, setIntegrand] = useState("");
  const [variable, setVariable] = useState("x");
  const [parameter, setParameter] = useState("a");
  const [lower, setLower] = useState("0");
  const [upper, setUpper] = useState("oo");
  const [sign, setSign] = useState<Sign>("positive");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<TelescopingResult | null>(null);

  async function run() {
    const request = buildTelescopingRequest({
      integrand,
      variable,
      parameter,
      lower,
      upper,
      parameterSign: sign,
    });
    if (typeof request === "string") {
      setError(request);
      return;
    }
    setError(null);
    setBusy(true);
    try {
      setResult(await api.telescoping(request));
    } catch (e) {
      setResult(null);
      setError(e instanceof ApiError ? e.message : "The search failed.");
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
          / Creative telescoping
        </p>
        <h1 className="font-serif text-2xl">Creative telescoping</h1>
        <p className="mt-1 max-w-2xl text-sm text-muted">
          For an integral that depends on a parameter, Minerva searches for an operator L in the
          parameter and a certificate G with L f = dG/dx, proves that identity exactly, checks
          convergence and the boundary terms from what you declare, derives the differential
          equation of the integral, and evaluates it only if every condition and an independent
          check agree. The integrand must be a product of rational functions, powers and
          exponentials of rational functions.
        </p>
      </div>

      <MathInput value={integrand} onChange={setIntegrand} onSubmit={run} />
      <p className="-mt-3 text-xs text-muted">
        The integrand only, no \int or dx. Example: \exp(-a x^2) or x^n \exp(-a x)
      </p>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
        <label className="block text-sm">
          Variable
          <input className={FIELD} value={variable} onChange={(e) => setVariable(e.target.value)} />
        </label>
        <label className="block text-sm">
          Parameter
          <input
            className={FIELD}
            value={parameter}
            onChange={(e) => setParameter(e.target.value)}
          />
        </label>
        <label className="block text-sm">
          Lower bound
          <input
            className={FIELD}
            value={lower}
            onChange={(e) => setLower(e.target.value)}
            placeholder="0 or -oo"
          />
        </label>
        <label className="block text-sm">
          Upper bound
          <input
            className={FIELD}
            value={upper}
            onChange={(e) => setUpper(e.target.value)}
            placeholder="1 or oo"
          />
        </label>
        <label className="block text-sm">
          The parameter is
          <select className={FIELD} value={sign} onChange={(e) => setSign(e.target.value as Sign)}>
            <option value="positive">positive</option>
            <option value="negative">negative</option>
            <option value="real">real (sign unknown)</option>
            <option value="none">not declared</option>
          </select>
        </label>
      </div>

      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={run}
          disabled={busy}
          className="rounded bg-accent px-4 py-2 text-sm font-medium text-paper disabled:opacity-50"
        >
          {busy ? "Searching…" : "Find the relation"}
        </button>
        {busy && <span className="text-xs text-muted">This can take up to a minute.</span>}
      </div>
      {error && (
        <p role="alert" className="rounded bg-bad-soft px-3 py-2 text-sm text-bad">
          {error}
        </p>
      )}
      {result && <TelescopingCard result={result} />}
    </div>
  );
}
