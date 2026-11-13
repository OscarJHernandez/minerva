"use client";

import Link from "next/link";
import { useState } from "react";
import { ApiError, api, type ConservedInput } from "@/lib/api";
import type { DiscoveryReport } from "@/lib/chat-types";
import { DiscoveryCard } from "./chat/discovery-card";
import { MathInput } from "./math-input";

const NAME = /^[A-Za-z][A-Za-z0-9_]{0,31}$/;
const FIELD =
  "w-full min-w-0 rounded border border-rule bg-raised px-2 py-1.5 text-sm text-ink disabled:opacity-50";

type System = "lagrangian" | "hamiltonian";
type Budget = "quick" | "standard" | "thorough";

function names(text: string): string[] {
  return text
    .split(/[\s,]+/)
    .map((s) => s.trim())
    .filter(Boolean);
}

/** Validates the form; returns the request or a message. Exported for tests. */
export function buildConservedRequest(form: {
  system: System;
  fn: string;
  coordinates: string;
  momenta: string;
  budget: Budget;
  advanced: boolean;
  degree: string;
  power: string;
  harmonics: string;
  timeDegree: string;
}): ConservedInput | string {
  if (!form.fn.trim()) return "Enter the Lagrangian or Hamiltonian.";
  const coordinates = names(form.coordinates);
  if (coordinates.length < 1 || coordinates.length > 3)
    return "Give between 1 and 3 coordinates, separated by commas.";
  if (!coordinates.every((c) => NAME.test(c))) return "Coordinate names are letters and digits.";
  let momenta: string[] | null = null;
  if (form.system === "hamiltonian") {
    momenta = names(form.momenta);
    if (momenta.length === 0)
      momenta = null; // default p_<coordinate>
    else if (momenta.length !== coordinates.length || !momenta.every((m) => NAME.test(m)))
      return "Give one momentum name per coordinate, or leave it empty for p_<coordinate>.";
  }
  const request: ConservedInput = {
    function: form.fn,
    system: form.system,
    coordinates,
    momenta,
    budget: form.budget,
  };
  if (form.advanced) {
    const num = (s: string) => (s.trim() === "" ? null : Number(s));
    const pins = {
      degree: num(form.degree),
      power_bound: num(form.power),
      harmonics: num(form.harmonics),
      time_degree: num(form.timeDegree),
    };
    if (Object.values(pins).some((v) => v !== null && !Number.isInteger(v)))
      return "Family limits are whole numbers.";
    if (Object.values(pins).some((v) => v !== null)) request.pins = pins;
  }
  return request;
}

/** Find conserved quantities: give L or H, optionally a budget; the search chooses its own families. */
export function ConservedPanel() {
  const [system, setSystem] = useState<System>("lagrangian");
  const [fn, setFn] = useState("");
  const [coordinates, setCoordinates] = useState("");
  const [momenta, setMomenta] = useState("");
  const [budget, setBudget] = useState<Budget>("standard");
  const [advanced, setAdvanced] = useState(false);
  const [degree, setDegree] = useState("");
  const [power, setPower] = useState("");
  const [harmonics, setHarmonics] = useState("");
  const [timeDegree, setTimeDegree] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<DiscoveryReport | null>(null);

  async function run() {
    const request = buildConservedRequest({
      system,
      fn,
      coordinates,
      momenta,
      budget,
      advanced,
      degree,
      power,
      harmonics,
      timeDegree,
    });
    if (typeof request === "string") {
      setError(request);
      return;
    }
    setError(null);
    setBusy(true);
    try {
      setReport(await api.conserved(request));
    } catch (e) {
      setReport(null);
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
          / Conserved quantities
        </p>
        <h1 className="font-serif text-2xl">Find conserved quantities</h1>
        <p className="mt-1 max-w-2xl text-sm text-muted">
          Give a Lagrangian or Hamiltonian. Minerva derives the equations of motion, searches larger
          and larger families of candidates, verifies each on the equations of motion, and reports
          which are functionally independent and exactly what it searched. You do not choose an
          ansatz; the answer is complete only within the families listed.
        </p>
      </div>

      <fieldset className="flex gap-4 text-sm">
        <legend className="sr-only">System</legend>
        {(["lagrangian", "hamiltonian"] as const).map((s) => (
          <label key={s} className="flex items-center gap-1.5">
            <input
              type="radio"
              name="system"
              checked={system === s}
              onChange={() => setSystem(s)}
            />
            {s === "lagrangian" ? "Lagrangian L" : "Hamiltonian H"}
          </label>
        ))}
      </fieldset>

      <MathInput value={fn} onChange={setFn} onSubmit={run} />
      <p className="-mt-3 text-xs text-muted">
        {system === "lagrangian"
          ? "Velocities as \\dot{q}. Example: \\frac12 m(\\dot{r}^2 + r^2\\dot{\\theta}^2) + \\frac{k}{r}"
          : "Momenta as p_x, p_\\theta. Example: \\frac{p_r^2}{2m} + \\frac{p_\\theta^2}{2mr^2} - \\frac{k}{r}"}
      </p>

      <div className="grid gap-3 sm:grid-cols-2">
        <label className="block text-sm">
          Coordinates (1 to 3, comma separated)
          <input
            className={FIELD}
            value={coordinates}
            onChange={(e) => setCoordinates(e.target.value)}
            placeholder="r, theta"
          />
        </label>
        {system === "hamiltonian" && (
          <label className="block text-sm">
            Momenta, in the same order (optional)
            <input
              className={FIELD}
              value={momenta}
              onChange={(e) => setMomenta(e.target.value)}
              placeholder="p_r, p_theta"
            />
          </label>
        )}
        <label className="block text-sm">
          Search budget
          <select
            className={FIELD}
            value={budget}
            onChange={(e) => setBudget(e.target.value as Budget)}
          >
            <option value="quick">Quick (small families, stops early)</option>
            <option value="standard">Standard</option>
            <option value="thorough">Thorough (adds powers of t)</option>
          </select>
        </label>
      </div>

      <details open={advanced} onToggle={(e) => setAdvanced(e.currentTarget.open)}>
        <summary className="cursor-pointer text-sm text-accent">
          Advanced: search exactly one family
        </summary>
        <div className="mt-2 grid grid-cols-2 gap-3 sm:grid-cols-4">
          {(
            [
              ["Momentum degree (1–4)", degree, setDegree],
              ["Power bound (0–4)", power, setPower],
              ["Harmonics (1–2)", harmonics, setHarmonics],
              ["Powers of t (0–2)", timeDegree, setTimeDegree],
            ] as const
          ).map(([label, value, set]) => (
            <label key={label} className="block text-xs text-muted">
              {label}
              <input
                className={FIELD}
                inputMode="numeric"
                value={value}
                onChange={(e) => set(e.target.value)}
              />
            </label>
          ))}
        </div>
        <p className="mt-1 text-xs text-muted">
          Leave empty to let the search expand by itself. Setting any of these replaces the ladder
          by a single family.
        </p>
      </details>

      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={run}
          disabled={busy}
          className="rounded bg-accent px-4 py-2 text-sm font-medium text-paper disabled:opacity-50"
        >
          {busy ? "Searching…" : "Find conserved quantities"}
        </button>
        {busy && <span className="text-xs text-muted">This can take up to a minute.</span>}
      </div>
      {error && (
        <p role="alert" className="rounded bg-bad-soft px-3 py-2 text-sm text-bad">
          {error}
        </p>
      )}
      {report && <DiscoveryCard report={report} />}
    </div>
  );
}
