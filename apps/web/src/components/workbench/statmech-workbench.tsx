"use client";

import { useEffect, useState } from "react";
import { ApiError, api } from "@/lib/api";
import {
  BASE_QUANTITIES,
  EXAMPLES,
  STATISTICS,
  blankForm,
  buildRequest,
  restoreForm,
  type StatForm,
} from "@/lib/statmech-form";
import type {
  StatAction,
  StatAnswers,
  StatKind,
  StatMechRequest,
  StatMechResult,
  StatQuantityName,
} from "@/lib/statmech-types";
import { AssumptionsEditor } from "../assumptions-editor";
import { StatMechPanel } from "../statmech-panel";

const FIELD =
  "min-w-0 rounded border border-rule bg-raised px-2 py-1.5 font-mono text-sm text-ink disabled:opacity-50";
const BUTTON =
  "rounded border border-rule bg-raised px-3 py-1.5 text-sm hover:bg-sunk disabled:opacity-50";

const ACTIONS: { id: StatAction; label: string }[] = [
  { id: "analyze", label: "Which methods apply?" },
  { id: "partition", label: "Partition function" },
  { id: "derive", label: "Derive quantities" },
  { id: "expectation", label: "Thermal average" },
  { id: "evaluate", label: "Evaluate numbers" },
];

const KINDS: { id: StatKind; label: string }[] = [
  { id: "levels", label: "List of levels" },
  { id: "level_family", label: "Level family E(n)" },
  { id: "density_of_states", label: "Density of states" },
  { id: "classical", label: "Classical Hamiltonian" },
  { id: "quantum", label: "Quantum (matrix H)" },
];

interface Run {
  key: number;
  action: StatAction;
  request: StatMechRequest;
  result: StatMechResult;
}

function Labelled({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex min-w-0 flex-col gap-1 text-xs text-muted">
      {label}
      {children}
    </label>
  );
}

/**
 * Define a statistical-mechanics problem once (system, conventions, assumptions) and carry it through
 * the chain: spectrum, partition function, free energy, internal energy, heat capacity. Every result is
 * checked by the server against an independent evaluation; no language model is involved. The
 * conventions that matter (phase-space cell, indistinguishable copies, numbers) are fields here.
 */
export function StatMechWorkbench({
  restore,
  onUnauthorized,
  onSaved,
}: {
  restore: { request: StatMechRequest; nonce: number } | null;
  onUnauthorized: () => void;
  onSaved: () => void;
}) {
  const [form, setForm] = useState<StatForm>(() => blankForm());
  const [runs, setRuns] = useState<Run[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!restore) return;
    // eslint-disable-next-line react-hooks/set-state-in-effect -- loading a saved problem into the form
    setForm(restoreForm(restore.request));
    setRuns([]);
    setError(null);
  }, [restore]);

  const set = <K extends keyof StatForm>(key: K, value: StatForm[K]) =>
    setForm((f) => ({ ...f, [key]: value }));

  async function execute(request: StatMechRequest) {
    setBusy(true);
    setError(null);
    try {
      const result = await api.statmech(request);
      setRuns((r) => [
        ...r,
        { key: (r.at(-1)?.key ?? 0) + 1, action: request.action, request, result },
      ]);
      onSaved();
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) onUnauthorized();
      else setError(e instanceof ApiError ? e.message : "The request failed. Try again.");
    } finally {
      setBusy(false);
    }
  }

  function run(action: StatAction, patch?: Partial<StatForm>) {
    const built = buildRequest(patch ? { ...form, ...patch } : form, action);
    if (!built.ok) {
      setError(built.error);
      return;
    }
    void execute(built.request);
  }

  /** The user answers a convention question on a result: the form keeps the answer, and it runs again. */
  function answer(action: StatAction, a: StatAnswers) {
    const next: StatForm = {
      ...form,
      classical:
        a.phase_space_cell !== undefined
          ? { ...form.classical, cell: a.phase_space_cell }
          : form.classical,
      copies:
        a.identical !== undefined
          ? { ...form.copies, identical: a.identical ? "yes" : "no" }
          : form.copies,
    };
    setForm(next);
    run(action, next);
  }

  const toggleQuantity = (id: StatQuantityName) =>
    set(
      "quantities",
      form.quantities.includes(id)
        ? form.quantities.filter((q) => q !== id)
        : [...form.quantities, id],
    );

  return (
    <div className="min-w-0 space-y-4" data-testid="statmech-workbench">
      <section
        aria-labelledby="statmech-form-heading"
        className="space-y-4 rounded-lg border border-rule bg-raised p-4"
      >
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 id="statmech-form-heading" className="font-serif text-xl">
            Statistical mechanics
          </h2>
          <label className="flex min-w-0 max-w-full items-center gap-2 text-xs text-muted">
            Start from
            <select
              aria-label="Start from an example"
              className={`${FIELD} max-w-full`}
              value=""
              onChange={(e) => {
                const chosen = EXAMPLES.find((x) => x.id === e.target.value);
                if (chosen) {
                  setForm(chosen.form);
                  setRuns([]);
                  setError(null);
                }
              }}
            >
              <option value="">an example…</option>
              {EXAMPLES.map((x) => (
                <option key={x.id} value={x.id}>
                  {x.label}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div role="group" aria-label="Kind of system" className="flex flex-wrap gap-2">
          {KINDS.map((k) => (
            <button
              key={k.id}
              type="button"
              aria-pressed={form.kind === k.id}
              onClick={() => set("kind", k.id)}
              className={`rounded border px-3 py-1.5 text-sm ${
                form.kind === k.id
                  ? "border-accent bg-accent text-accent-ink"
                  : "border-rule bg-raised hover:bg-sunk"
              }`}
            >
              {k.label}
            </button>
          ))}
        </div>

        {form.kind === "levels" && (
          <div className="space-y-2">
            {form.levels.map((l, i) => (
              <div key={i} className="grid grid-cols-[1fr_6rem_auto] items-end gap-2">
                <Labelled label={`Energy of level ${i + 1}`}>
                  <input
                    className={FIELD}
                    value={l.energy}
                    onChange={(e) =>
                      set(
                        "levels",
                        form.levels.map((x, j) => (j === i ? { ...x, energy: e.target.value } : x)),
                      )
                    }
                  />
                </Labelled>
                <Labelled label="Degeneracy">
                  <input
                    className={FIELD}
                    value={l.degeneracy}
                    onChange={(e) =>
                      set(
                        "levels",
                        form.levels.map((x, j) =>
                          j === i ? { ...x, degeneracy: e.target.value } : x,
                        ),
                      )
                    }
                  />
                </Labelled>
                <button
                  type="button"
                  className={BUTTON}
                  aria-label={`Remove level ${i + 1}`}
                  disabled={form.levels.length <= 1}
                  onClick={() =>
                    set(
                      "levels",
                      form.levels.filter((_, j) => j !== i),
                    )
                  }
                >
                  ×
                </button>
              </div>
            ))}
            <button
              type="button"
              className={BUTTON}
              disabled={form.levels.length >= 64}
              onClick={() => set("levels", [...form.levels, { energy: "", degeneracy: "1" }])}
            >
              Add a level
            </button>
          </div>
        )}

        {form.kind === "level_family" && (
          <div className="grid gap-2 sm:grid-cols-2">
            <Labelled label="Energy E(n)">
              <input
                className={FIELD}
                value={form.family.energy}
                placeholder="hbar*omega*(n + 1/2)"
                onChange={(e) => set("family", { ...form.family, energy: e.target.value })}
              />
            </Labelled>
            <Labelled label="Degeneracy g(n)">
              <input
                className={FIELD}
                value={form.family.degeneracy}
                onChange={(e) => set("family", { ...form.family, degeneracy: e.target.value })}
              />
            </Labelled>
            <Labelled label="Index name">
              <input
                className={FIELD}
                value={form.family.index}
                onChange={(e) => set("family", { ...form.family, index: e.target.value })}
              />
            </Labelled>
            <div className="grid grid-cols-2 gap-2">
              <Labelled label="From">
                <input
                  className={FIELD}
                  value={form.family.lower}
                  onChange={(e) => set("family", { ...form.family, lower: e.target.value })}
                />
              </Labelled>
              <Labelled label="To (oo for infinity)">
                <input
                  className={FIELD}
                  value={form.family.upper}
                  onChange={(e) => set("family", { ...form.family, upper: e.target.value })}
                />
              </Labelled>
            </div>
          </div>
        )}

        {form.kind === "density_of_states" && (
          <div className="grid gap-2 sm:grid-cols-2">
            <Labelled label="Density of states ρ(En)">
              <input
                className={FIELD}
                value={form.dos.density}
                placeholder="sqrt(En)"
                onChange={(e) => set("dos", { ...form.dos, density: e.target.value })}
              />
            </Labelled>
            <Labelled label="Energy variable (E is Euler's number in text syntax)">
              <input
                className={FIELD}
                value={form.dos.variable}
                onChange={(e) => set("dos", { ...form.dos, variable: e.target.value })}
              />
            </Labelled>
            <Labelled label="Lowest energy">
              <input
                className={FIELD}
                value={form.dos.lower}
                onChange={(e) => set("dos", { ...form.dos, lower: e.target.value })}
              />
            </Labelled>
            <Labelled label="Highest energy (oo for none)">
              <input
                className={FIELD}
                value={form.dos.upper}
                onChange={(e) => set("dos", { ...form.dos, upper: e.target.value })}
              />
            </Labelled>
          </div>
        )}

        {form.kind === "quantum" && (
          <div className="space-y-2">
            <Labelled label="Hamiltonian matrix H: one row per line, entries separated by ; (I is the imaginary unit)">
              <textarea
                className={FIELD}
                rows={3}
                value={form.quantum.hamiltonian}
                onChange={(e) => set("quantum", { ...form.quantum, hamiltonian: e.target.value })}
              />
            </Labelled>
            <div className="grid gap-2 sm:grid-cols-[8rem_1fr]">
              <Labelled label="Operator name">
                <input
                  className={FIELD}
                  value={form.quantum.observableName}
                  onChange={(e) =>
                    set("quantum", { ...form.quantum, observableName: e.target.value })
                  }
                />
              </Labelled>
              <Labelled label="Operator matrix (optional; never assumed to commute with H)">
                <textarea
                  className={FIELD}
                  rows={3}
                  value={form.quantum.observable}
                  onChange={(e) => set("quantum", { ...form.quantum, observable: e.target.value })}
                />
              </Labelled>
            </div>
            <p className="text-xs text-muted">
              Declare the parameters real (or positive) so that H is provably Hermitian. Use the
              operator name as the observable under “Thermal average options”.
            </p>
          </div>
        )}

        {form.kind === "classical" && (
          <div className="space-y-2">
            <Labelled label="Hamiltonian of one copy, as a sum of one-variable terms">
              <input
                className={FIELD}
                value={form.classical.hamiltonian}
                placeholder="p**2/(2*m) + m*omega**2*x**2/2"
                onChange={(e) =>
                  set("classical", { ...form.classical, hamiltonian: e.target.value })
                }
              />
            </Labelled>
            {form.classical.variables.map((v, i) => (
              <div key={i} className="grid grid-cols-3 items-end gap-2">
                <Labelled label={`Variable ${i + 1}`}>
                  <input
                    className={FIELD}
                    value={v.name}
                    onChange={(e) =>
                      set("classical", {
                        ...form.classical,
                        variables: form.classical.variables.map((x, j) =>
                          j === i ? { ...x, name: e.target.value } : x,
                        ),
                      })
                    }
                  />
                </Labelled>
                <Labelled label="Domain from">
                  <input
                    className={FIELD}
                    value={v.lower}
                    onChange={(e) =>
                      set("classical", {
                        ...form.classical,
                        variables: form.classical.variables.map((x, j) =>
                          j === i ? { ...x, lower: e.target.value } : x,
                        ),
                      })
                    }
                  />
                </Labelled>
                <Labelled label="to">
                  <input
                    className={FIELD}
                    value={v.upper}
                    onChange={(e) =>
                      set("classical", {
                        ...form.classical,
                        variables: form.classical.variables.map((x, j) =>
                          j === i ? { ...x, upper: e.target.value } : x,
                        ),
                      })
                    }
                  />
                </Labelled>
              </div>
            ))}
            <button
              type="button"
              className={BUTTON}
              onClick={() =>
                set("classical", {
                  ...form.classical,
                  variables: [...form.classical.variables, { name: "", lower: "-oo", upper: "oo" }],
                })
              }
            >
              Add a variable
            </button>
            <Labelled label="Phase-space cell (your choice, e.g. h^3, or 1)">
              <input
                className={FIELD}
                value={form.classical.cell}
                onChange={(e) => set("classical", { ...form.classical, cell: e.target.value })}
              />
            </Labelled>
          </div>
        )}

        <div className="flex flex-wrap items-end gap-4">
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={form.copies.on}
              onChange={(e) => set("copies", { ...form.copies, on: e.target.checked })}
            />
            Several independent copies
          </label>
          {form.copies.on && (
            <>
              <Labelled label="Number of copies">
                <input
                  className={`${FIELD} w-24`}
                  value={form.copies.count}
                  onChange={(e) => set("copies", { ...form.copies, count: e.target.value })}
                />
              </Labelled>
              <fieldset className="text-sm">
                <legend className="text-xs text-muted">The copies are</legend>
                <label className="mr-3">
                  <input
                    type="radio"
                    name="identical"
                    checked={form.copies.identical === "yes"}
                    onChange={() => set("copies", { ...form.copies, identical: "yes" })}
                  />{" "}
                  indistinguishable (÷ N!)
                </label>
                <label>
                  <input
                    type="radio"
                    name="identical"
                    checked={form.copies.identical === "no"}
                    onChange={() => set("copies", { ...form.copies, identical: "no" })}
                  />{" "}
                  distinguishable
                </label>
              </fieldset>
            </>
          )}
        </div>

        <div className="grid gap-2 sm:grid-cols-3">
          <Labelled label="Write results in">
            <select
              className={FIELD}
              value={form.temperature}
              onChange={(e) => set("temperature", e.target.value as "beta" | "T")}
            >
              <option value="beta">β (inverse temperature)</option>
              <option value="T">T (with k_B)</option>
            </select>
          </Labelled>
          <Labelled label="Other state variables, comma-separated (V, h, …)">
            <input
              className={FIELD}
              value={form.parameters}
              onChange={(e) => set("parameters", e.target.value)}
            />
          </Labelled>
        </div>

        <AssumptionsEditor value={form.assumptions} onChange={(a) => set("assumptions", a)} />

        <fieldset className="space-y-2">
          <legend className="text-sm font-medium">Quantities to derive</legend>
          <div className="flex flex-wrap gap-3">
            {BASE_QUANTITIES.map((q) => (
              <label key={q.id} className="flex items-center gap-1.5 text-sm">
                <input
                  type="checkbox"
                  checked={form.quantities.includes(q.id)}
                  onChange={() => toggleQuantity(q.id)}
                />
                {q.label}
              </label>
            ))}
          </div>
          {form.partials.map((p, i) => (
            <div key={i} className="grid grid-cols-2 items-end gap-2 sm:grid-cols-6">
              <Labelled label="Derivative of">
                <select
                  className={FIELD}
                  value={p.of}
                  onChange={(e) =>
                    set(
                      "partials",
                      form.partials.map((x, j) =>
                        j === i ? { ...x, of: e.target.value as StatQuantityName } : x,
                      ),
                    )
                  }
                >
                  {BASE_QUANTITIES.map((q) => (
                    <option key={q.id} value={q.id}>
                      {q.id}
                    </option>
                  ))}
                </select>
              </Labelled>
              {(
                [
                  ["variable", "with respect to"],
                  ["held", "held fixed"],
                  ["scale", "times"],
                  ["label", "call it"],
                ] as const
              ).map(([field, label]) => (
                <Labelled key={field} label={label}>
                  <input
                    className={FIELD}
                    value={p[field]}
                    onChange={(e) =>
                      set(
                        "partials",
                        form.partials.map((x, j) =>
                          j === i ? { ...x, [field]: e.target.value } : x,
                        ),
                      )
                    }
                  />
                </Labelled>
              ))}
              <button
                type="button"
                className={BUTTON}
                aria-label={`Remove derivative ${i + 1}`}
                onClick={() =>
                  set(
                    "partials",
                    form.partials.filter((_, j) => j !== i),
                  )
                }
              >
                ×
              </button>
            </div>
          ))}
          <button
            type="button"
            className={BUTTON}
            onClick={() =>
              set("partials", [
                ...form.partials,
                { of: "U", variable: "T", held: "", scale: "1", label: "" },
              ])
            }
          >
            Add a partial derivative (heat capacity, pressure, …)
          </button>
        </fieldset>

        <details className="rounded border border-rule p-3">
          <summary className="cursor-pointer text-sm font-medium">Thermal average options</summary>
          <div className="mt-2 grid gap-2 sm:grid-cols-3">
            <Labelled label="Observable (E or a name given per level)">
              <input
                className={FIELD}
                value={form.observable}
                onChange={(e) => set("observable", e.target.value)}
              />
            </Labelled>
            <Labelled label="Statistic">
              <select
                className={FIELD}
                value={form.statistic}
                onChange={(e) => set("statistic", e.target.value as typeof form.statistic)}
              >
                {STATISTICS.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.label}
                  </option>
                ))}
              </select>
            </Labelled>
            <Labelled label="Order (1 to 4)">
              <input
                className={FIELD}
                type="number"
                min={1}
                max={4}
                value={form.order}
                onChange={(e) => set("order", Number(e.target.value) || 1)}
              />
            </Labelled>
          </div>
        </details>

        <details className="rounded border border-rule p-3">
          <summary className="cursor-pointer text-sm font-medium">
            Numbers for a numerical evaluation (yours)
          </summary>
          <div className="mt-2 space-y-2">
            <Labelled label="One value per line, e.g. beta = 1 (give k_B too, e.g. k_B = 1 in reduced units)">
              <textarea
                className={FIELD}
                rows={4}
                value={form.values}
                onChange={(e) => set("values", e.target.value)}
              />
            </Labelled>
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={form.sweep.on}
                onChange={(e) => set("sweep", { ...form.sweep, on: e.target.checked })}
              />
              Sweep one variable
            </label>
            {form.sweep.on && (
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-5">
                {(
                  [
                    ["variable", "Variable"],
                    ["start", "From"],
                    ["stop", "To"],
                    ["points", "Points"],
                  ] as const
                ).map(([field, label]) => (
                  <Labelled key={field} label={label}>
                    <input
                      className={FIELD}
                      value={form.sweep[field]}
                      onChange={(e) => set("sweep", { ...form.sweep, [field]: e.target.value })}
                    />
                  </Labelled>
                ))}
                <Labelled label="Spacing">
                  <select
                    className={FIELD}
                    value={form.sweep.scale}
                    onChange={(e) =>
                      set("sweep", { ...form.sweep, scale: e.target.value as "linear" | "log" })
                    }
                  >
                    <option value="linear">linear</option>
                    <option value="log">logarithmic</option>
                  </select>
                </Labelled>
              </div>
            )}
          </div>
        </details>

        <div className="flex flex-wrap gap-2">
          {ACTIONS.map((a) => (
            <button
              key={a.id}
              type="button"
              className={`${BUTTON} ${a.id === "derive" ? "border-accent" : ""}`}
              disabled={busy}
              onClick={() => run(a.id)}
            >
              {a.label}
            </button>
          ))}
        </div>
        {error && (
          <p role="alert" className="text-sm text-bad">
            {error}
          </p>
        )}
      </section>

      {runs
        .slice()
        .reverse()
        .map((r) => (
          <StatMechPanel
            key={r.key}
            result={r.result}
            needs={{
              cell: r.request.system.kind === "classical" && !r.request.system.phase_space_cell,
              identical: r.request.copies != null && r.request.copies.identical === null,
            }}
            onAnswer={(a) => answer(r.action, a)}
          />
        ))}
    </div>
  );
}
