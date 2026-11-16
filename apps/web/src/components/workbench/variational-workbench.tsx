"use client";

import { useFormDraft } from "@/lib/use-form-draft";
import { useEffect, useId, useState } from "react";
import { ApiError, api } from "@/lib/api";
import type { HistoryItem } from "@/lib/types";
import {
  EXAMPLES,
  blankForm,
  buildVariationRequest,
  formFromSteps,
  restoreVariationForm,
  type VariationForm,
} from "@/lib/variation-form";
import type { VariationAction, VariationReport, VariationRequest } from "@/lib/variation-types";
import { AssumptionsEditor } from "../assumptions-editor";
import { ConditionsEditor, FIELD, NumbersFields } from "../variation-fields";
import { VariationPanel } from "../variation-panel";

/** A saved derivation (or a blank one) to load into the workbench; the nonce lets it be loaded again. */
export interface VariationRestore {
  nonce: number;
  derivationId: string | null;
  steps: HistoryItem[];
  /** Only the functional (a saved single calculation): fills the form, with no steps shown. */
  request?: Partial<VariationRequest>;
}

const BUTTON =
  "rounded border border-accent px-3 py-1.5 text-sm font-medium text-accent hover:bg-accent-soft disabled:opacity-50";

const BUSY_TEXT: Partial<Record<VariationAction, string>> = {
  solve: "Searching for a stationary path and checking it…",
  verify: "Comparing with finite differences…",
  evaluate: "Evaluating the action…",
};

/**
 * Variational calculus as a derivation. The functional stays in the form; each action appends a step
 * (functional, first variation, boundary terms, numerical checks, solution) to the same saved
 * derivation. The server returns a result only when its own independent numerical checks pass, and
 * nothing here involves a language model. The numbers (interval, conditions' values, parameters,
 * paths) are only what the user typed.
 */
export function VariationalWorkbench({
  restore,
  draftKey,
  onUnauthorized,
  onSaved,
}: {
  draftKey?: string;
  restore: VariationRestore | null;
  onUnauthorized: () => void;
  onSaved: () => void;
}) {
  const uid = useId();
  const [form, setForm] = useState<VariationForm>(() => blankForm());
  useFormDraft(form, setForm, draftKey, restore !== null);
  const [derivationId, setDerivationId] = useState<string | null>(null);
  const [steps, setSteps] = useState<HistoryItem[]>([]);
  const [busy, setBusy] = useState<VariationAction | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!restore) return;
    /* eslint-disable react-hooks/set-state-in-effect -- loading a saved derivation into the form */
    setForm(
      restore.steps.length > 0
        ? formFromSteps(restore.steps)
        : restore.request
          ? restoreVariationForm(restore.request)
          : blankForm(),
    );
    setDerivationId(restore.derivationId);
    setSteps(restore.steps);
    setError(null);
    /* eslint-enable react-hooks/set-state-in-effect */
  }, [restore]);

  const reports = steps.filter((s) => s.operation === "variation");
  const parameters =
    [...reports]
      .reverse()
      .map((s) => (s.result as unknown as VariationReport).analysis?.parameters)
      .find((p) => p !== undefined) ?? [];

  function newDerivation(next: VariationForm) {
    setForm(next);
    setDerivationId(null);
    setSteps([]);
    setError(null);
  }

  async function run(action: VariationAction) {
    if (busy) return;
    // the button asks for the symbolic second variation; a path is an input of the numerical steps
    const built = buildVariationRequest(
      action === "second" ? { ...form, paths: {} } : form,
      action,
    );
    if (!built.ok) {
      setError(built.error);
      return;
    }
    setBusy(action);
    setError(null);
    try {
      let id = derivationId;
      if (!id) {
        id = (await api.createDerivation()).id;
        setDerivationId(id);
      }
      await api.variation(built.request, { derivation_id: id, parent_id: steps.at(-1)?.id });
      setSteps((await api.derivation(id)).steps);
      onSaved();
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) onUnauthorized();
      else setError(e instanceof ApiError ? e.message : "The request failed. Try again.");
    } finally {
      setBusy(null);
    }
  }

  async function deleteStep(item: HistoryItem) {
    try {
      await api.deleteHistoryItem(item.id);
      if (derivationId) setSteps((await api.derivation(derivationId)).steps);
      onSaved();
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) onUnauthorized();
      else setError(e instanceof ApiError ? e.message : "The step could not be deleted.");
    }
  }

  const set = (update: (f: VariationForm) => VariationForm) => setForm(update);
  const formatLabel = form.format === "latex" ? "LaTeX" : "Plain text";

  return (
    <div className="min-w-0 space-y-4" data-testid="variational-workbench">
      <section
        aria-labelledby={`${uid}-heading`}
        className="space-y-4 rounded-lg border border-rule bg-raised p-4"
      >
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 id={`${uid}-heading`} className="font-serif text-xl">
            Variational calculus
          </h2>
          <label className="flex min-w-0 max-w-full items-center gap-2 text-xs text-muted">
            Start from
            <select
              aria-label="Start from an example"
              className={`${FIELD} min-w-0 flex-1`}
              value=""
              onChange={(e) => {
                const chosen = EXAMPLES.find((x) => x.id === e.target.value);
                if (chosen) newDerivation(chosen.form);
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
        <p className="max-w-prose text-sm text-muted">
          Give the integrand of S[x] = ∫ L dt and name the unknown functions. Each action below adds
          a step to this derivation; start a new derivation to change the problem. Endpoint
          conditions are only what you declare: a derivative you do not list stays undecided and its
          boundary term is kept.
        </p>

        <div className="space-y-2">
          <div className="flex flex-wrap items-center gap-3">
            <label className="text-xs uppercase tracking-wide text-muted" htmlFor={`${uid}-L`}>
              Integrand L
            </label>
            <select
              aria-label="Input syntax"
              className={FIELD}
              value={form.format}
              onChange={(e) => set((f) => ({ ...f, format: e.target.value as "latex" | "text" }))}
            >
              <option value="latex">LaTeX</option>
              <option value="text">Plain text</option>
            </select>
          </div>
          <input
            id={`${uid}-L`}
            className={`${FIELD} block w-full`}
            placeholder={
              form.format === "latex" ? "\\frac{m}{2}\\dot{x}^2-V(x)" : "m/2*x_dot**2 - V(x)"
            }
            value={form.integrand}
            onChange={(e) => set((f) => ({ ...f, integrand: e.target.value }))}
          />
          <p className="text-xs text-muted">
            {formatLabel}. Derivatives: \dot x, \ddot x, \dddot x, x^(4).
          </p>
        </div>

        <div className="grid gap-3 sm:grid-cols-3">
          <label className="text-xs text-muted">
            Variable
            <input
              aria-label="Variable"
              className={`${FIELD} mt-0.5 block w-full`}
              value={form.variable}
              onChange={(e) => set((f) => ({ ...f, variable: e.target.value }))}
            />
          </label>
          <label className="text-xs text-muted sm:col-span-1">
            Unknown functions, separated by commas
            <input
              aria-label="Unknown functions"
              className={`${FIELD} mt-0.5 block w-full`}
              value={form.functions}
              onChange={(e) => set((f) => ({ ...f, functions: e.target.value }))}
            />
          </label>
          <label className="text-xs text-muted">
            Given functions, e.g. V(x)
            <input
              aria-label="Given functions"
              className={`${FIELD} mt-0.5 block w-full`}
              value={form.givenFunctions}
              onChange={(e) => set((f) => ({ ...f, givenFunctions: e.target.value }))}
            />
          </label>
        </div>

        <div className="grid gap-3 sm:grid-cols-2">
          <label className="text-xs text-muted">
            Lower end
            <input
              aria-label="Lower end"
              className={`${FIELD} mt-0.5 block w-full`}
              placeholder="a, or a number"
              value={form.lower}
              onChange={(e) => set((f) => ({ ...f, lower: e.target.value }))}
            />
          </label>
          <label className="text-xs text-muted">
            Upper end
            <input
              aria-label="Upper end"
              className={`${FIELD} mt-0.5 block w-full`}
              placeholder="b, or a number"
              value={form.upper}
              onChange={(e) => set((f) => ({ ...f, upper: e.target.value }))}
            />
          </label>
          <label className="text-xs text-muted">
            Term of the action at the lower end (optional)
            <input
              aria-label="Lower endpoint term"
              className={`${FIELD} mt-0.5 block w-full`}
              value={form.lowerTerm}
              onChange={(e) => set((f) => ({ ...f, lowerTerm: e.target.value }))}
            />
          </label>
          <label className="text-xs text-muted">
            Term of the action at the upper end (optional)
            <input
              aria-label="Upper endpoint term"
              className={`${FIELD} mt-0.5 block w-full`}
              value={form.upperTerm}
              onChange={(e) => set((f) => ({ ...f, upperTerm: e.target.value }))}
            />
          </label>
        </div>

        <ConditionsEditor form={form} setForm={set} />

        <AssumptionsEditor
          value={form.assumptions}
          onChange={(assumptions) => set((f) => ({ ...f, assumptions }))}
        />

        <div className="flex flex-wrap items-center gap-2">
          {(
            [
              ["analyze", "Analyze"],
              ["first", "First variation"],
              ["second", "Second variation"],
            ] as const
          ).map(([action, label]) => (
            <button
              key={action}
              type="button"
              className={BUTTON}
              disabled={busy !== null}
              onClick={() => void run(action)}
            >
              {busy === action ? "Working…" : label}
            </button>
          ))}
        </div>

        <fieldset className="min-w-0 space-y-3 rounded border border-rule p-3">
          <legend className="px-1 text-xs uppercase tracking-wide text-muted">Numbers</legend>
          <p className="text-xs text-muted">
            For the numerical steps, the interval above, the values of the fixed conditions, the
            parameters and a path are yours to choose; none is filled in for you.
          </p>
          <NumbersFields form={form} setForm={set} parameters={parameters} />
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={form.withSecond}
              onChange={(e) => set((f) => ({ ...f, withSecond: e.target.checked }))}
            />
            Also classify the stationary path (second variation)
          </label>
          <div className="flex flex-wrap items-center gap-2">
            {(
              [
                ["evaluate", "Evaluate the action"],
                ["verify", "Verify by finite differences"],
                ["solve", "Find stationary path"],
              ] as const
            ).map(([action, label]) => (
              <button
                key={action}
                type="button"
                className={BUTTON}
                disabled={busy !== null}
                onClick={() => void run(action)}
              >
                {busy === action ? "Working…" : label}
              </button>
            ))}
          </div>
        </fieldset>

        {busy && BUSY_TEXT[busy] && (
          <p role="status" className="text-sm text-muted">
            {BUSY_TEXT[busy]}
          </p>
        )}
        {error && (
          <p role="alert" className="rounded bg-bad-soft px-3 py-2 text-sm text-bad">
            {error}
          </p>
        )}
      </section>

      {reports.length === 0 ? (
        <section
          aria-label="Derivation"
          className="rounded-lg border border-dashed border-rule p-6 text-sm text-muted"
        >
          Each action adds a step here, in order: the functional, the first variation, the boundary
          terms, the numerical checks and the stationary path. No language model is involved.
        </section>
      ) : (
        <section aria-label="Derivation">
          <h2 className="mb-3 font-serif text-xl">
            Derivation <span className="text-sm font-normal text-muted">({reports.length})</span>
          </h2>
          <ol className="space-y-3">
            {reports.map((item, i) => (
              <li key={item.id} id={`step-${item.id}`} aria-label={`Step ${i + 1}`}>
                <div className="mb-1 flex items-center justify-between text-xs text-muted">
                  <span>Step {i + 1}</span>
                  <button
                    type="button"
                    aria-label={`Delete step ${i + 1}`}
                    className="hover:text-bad"
                    onClick={() => void deleteStep(item)}
                  >
                    Delete
                  </button>
                </div>
                <VariationPanel
                  report={item.result as unknown as VariationReport}
                  headingId={`${uid}-step-${item.id}`}
                />
              </li>
            ))}
          </ol>
        </section>
      )}
    </div>
  );
}
