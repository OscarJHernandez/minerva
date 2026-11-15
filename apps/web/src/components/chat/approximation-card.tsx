"use client";

import { useState } from "react";
import { ApiError } from "@/lib/api";
import {
  buildToolArgs,
  restoreApproximationForm,
  type ApproximationForm,
} from "@/lib/approximation-form";
import type { ApproximationReport, ApproximationToolArgs } from "@/lib/approximation-types";
import { ApproximationFields } from "../approximation-fields";
import { ApproximationPanel } from "../approximation-panel";
import { useMotionRunner } from "./motion-runner";

/**
 * An approximation in a conversation, with a form for the numbers. The interval, point, order,
 * tolerance and period belong to the user: the form starts with what the user's message contained
 * and nothing else, and posts the expression exactly as the model sent it plus what is typed.
 */
export function ApproximationCard({
  report,
  request,
}: {
  report: ApproximationReport;
  request: ApproximationToolArgs;
}) {
  const run = useMotionRunner()?.approximation;
  const [form, setForm] = useState<ApproximationForm>(() => restoreApproximationForm(request));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!run) return;
    const built = buildToolArgs(request, form);
    if (!built.ok) {
      setError(built.error);
      return;
    }
    setError(null);
    setBusy(true);
    try {
      await run(built.args);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "The request could not be run.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <ApproximationPanel report={report} />
      {run && (
        <details
          className="mt-2 rounded-lg border border-rule bg-raised p-4"
          open={report.questions.length > 0}
        >
          <summary className="cursor-pointer text-sm font-medium">
            Run with your numbers (you choose them): show
          </summary>
          <form aria-label="Run with your numbers" className="mt-3 space-y-3" onSubmit={submit}>
            <ApproximationFields form={form} onChange={setForm} disabled={busy} />
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
              {busy ? "Approximating and checking…" : "Approximate"}
            </button>
          </form>
        </details>
      )}
    </div>
  );
}
