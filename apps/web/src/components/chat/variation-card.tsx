"use client";

import { useState } from "react";
import { ApiError } from "@/lib/api";
import {
  NUMERIC_ACTIONS,
  buildVariationRequest,
  restoreVariationForm,
  type VariationForm,
} from "@/lib/variation-form";
import type { FunctionalRequest, VariationAction, VariationReport } from "@/lib/variation-types";
import { ConditionsEditor, FIELD, NumbersFields } from "../variation-fields";
import { VariationPanel } from "../variation-panel";
import { useMotionRunner } from "./motion-runner";

const ACTION_TEXT: Record<string, string> = {
  evaluate: "Evaluate the action on my path",
  verify: "Verify by finite differences",
  solve: "Find a stationary path",
  second: "Second variation (symbolic)",
};

/**
 * The variational result in a conversation, with a form for the numerical steps. The numbers belong to
 * the user: the model cannot run these steps and nothing here is pre-filled. The form posts the
 * functional exactly as the model sent it, plus what the user typed and the chosen action.
 */
export function VariationCard({
  report,
  request,
}: {
  report: VariationReport;
  request: FunctionalRequest;
}) {
  const runner = useMotionRunner();
  const run = runner?.variation;
  const [form, setForm] = useState<VariationForm>(() => restoreVariationForm(request));
  const [action, setAction] = useState<VariationAction>("evaluate");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const shown = report.status === "ok" || report.status === "unchanged";
  const parameters = report.analysis?.parameters ?? [];

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!run) return;
    setError(null);
    const built = buildVariationRequest(form, action);
    if (!built.ok) return setError(built.error);
    setBusy(true);
    try {
      await run(built.request);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "That could not be run. Try again.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <VariationPanel report={report} />
      {run && shown && report.analysis && (
        <details className="mt-3 rounded border border-rule p-3">
          <summary className="cursor-pointer text-sm font-medium">
            Run with your numbers{" "}
            <span className="font-normal text-muted">(you choose them): show</span>
          </summary>
          <form onSubmit={submit} className="mt-3 space-y-4" aria-label="Run with your numbers">
            <div className="grid gap-3 sm:grid-cols-2">
              <label className="text-xs text-muted">
                Lower end
                <input
                  aria-label="Lower end"
                  className={`${FIELD} mt-0.5 block w-full`}
                  value={form.lower}
                  onChange={(e) => setForm((f) => ({ ...f, lower: e.target.value }))}
                />
              </label>
              <label className="text-xs text-muted">
                Upper end
                <input
                  aria-label="Upper end"
                  className={`${FIELD} mt-0.5 block w-full`}
                  value={form.upper}
                  onChange={(e) => setForm((f) => ({ ...f, upper: e.target.value }))}
                />
              </label>
            </div>
            <ConditionsEditor form={form} setForm={(u) => setForm(u)} />
            <NumbersFields form={form} setForm={(u) => setForm(u)} parameters={parameters} />
            <div className="flex flex-wrap items-center gap-3">
              <label className="text-xs text-muted">
                Action
                <select
                  aria-label="Action"
                  className={`${FIELD} ml-2`}
                  value={action}
                  onChange={(e) => setAction(e.target.value as VariationAction)}
                >
                  {[...NUMERIC_ACTIONS, "second" as const].map((a) => (
                    <option key={a} value={a}>
                      {ACTION_TEXT[a]}
                    </option>
                  ))}
                </select>
              </label>
              {action === "solve" && (
                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={form.withSecond}
                    onChange={(e) => setForm((f) => ({ ...f, withSecond: e.target.checked }))}
                  />
                  Also classify (second variation)
                </label>
              )}
            </div>
            <button
              type="submit"
              disabled={busy}
              className="rounded bg-accent px-4 py-2 text-sm font-medium text-accent-ink disabled:opacity-50"
            >
              {busy ? "Running and checking…" : "Run"}
            </button>
            {error && (
              <p role="alert" className="rounded bg-bad-soft px-3 py-2 text-sm text-bad">
                {error}
              </p>
            )}
          </form>
        </details>
      )}
    </div>
  );
}
