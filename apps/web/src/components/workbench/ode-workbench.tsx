"use client";

import { useEffect, useId, useState } from "react";
import { ApiError, api } from "@/lib/api";
import {
  BLANK_ODE_FORM,
  ODE_EXAMPLES,
  buildOdeInput,
  restoreOdeForm,
  type OdeForm,
} from "@/lib/ode-form";
import type {
  MethodSuggestion,
  OdeAnalysis,
  OdeAnalyzeInput,
  OdeInput,
  OdeNumericInput,
  OdeNumericReport,
  OdeSolution,
  OdeSolveInput,
} from "@/lib/ode-types";
import { AssumptionsEditor } from "../assumptions-editor";
import { OdeAnalysisPanel } from "../ode-analysis-panel";
import { OdeBoundaryForm, OdeNumericForm, OdeQualitativeForm, OdeSolveForm } from "../ode-forms";
import { OdeNumericPanel } from "../ode-numeric-panel";
import { OdeSolutionPanel } from "../ode-solution-panel";

const FIELD =
  "min-w-0 rounded border border-rule bg-raised px-2 py-1.5 font-mono text-sm text-ink disabled:opacity-50";
const ACTION =
  "rounded border border-rule px-2 py-1 text-xs hover:border-accent disabled:opacity-50";

type FormName = "ivp" | "bvp" | "series" | "perturbation" | "qualitative";

type Run =
  | { key: number; kind: "analysis"; request: OdeInput | OdeAnalyzeInput; report: OdeAnalysis }
  | { key: number; kind: "symbolic"; request: OdeInput; report: OdeSolution }
  | { key: number; kind: "numeric"; request: OdeNumericInput; report: OdeNumericReport };

type NewRun =
  | { kind: "analysis"; request: OdeInput | OdeAnalyzeInput; report: OdeAnalysis }
  | { kind: "symbolic"; request: OdeInput; report: OdeSolution }
  | { kind: "numeric"; request: OdeNumericInput; report: OdeNumericReport };

/**
 * Differential equations: write the equation, see how Minerva read it, see the approaches that can be
 * tried (ranked, with what each needs), choose one, supply the numbers it needs, and compare what the
 * methods give. No language model is involved. Every result carries its own checks; a numerical
 * trajectory is an approximation on the interval you gave, and a closed form is shown only after it was
 * substituted back into your equation.
 */
export function OdeWorkbench({
  restore,
  onUnauthorized,
  onSaved,
  onUseExpression,
}: {
  /** A saved problem to load into the form (the nonce lets the same one be loaded again). */
  restore: { request: OdeInput; nonce: number } | null;
  onUnauthorized: () => void;
  onSaved: () => void;
  /** Continue in Expression mode with a right-hand side (text syntax). */
  onUseExpression?: (text: string, equationLatex: string) => void;
}) {
  const [form, setForm] = useState<OdeForm>(BLANK_ODE_FORM);
  const [runs, setRuns] = useState<Run[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  /** Which analysis has a form open, and which one. */
  const [formFor, setFormFor] = useState<{ key: number; form: FormName } | null>(null);
  const uid = useId();

  useEffect(() => {
    if (!restore) return;
    // eslint-disable-next-line react-hooks/set-state-in-effect -- loading a saved problem into the form
    setForm(restoreOdeForm(restore.request));
    setRuns([]);
    setError(null);
    setFormFor(null);
  }, [restore]);

  function add(run: NewRun) {
    setRuns((all) => [...all, { ...run, key: (all.at(-1)?.key ?? 0) + 1 } as Run]);
    onSaved();
  }

  function fail(e: unknown, fallback: string) {
    if (e instanceof ApiError && e.status === 401) onUnauthorized();
    else setError(e instanceof ApiError ? e.message : fallback);
  }

  async function analyse() {
    const built = buildOdeInput(form);
    if (!built.ok) return setError(built.error);
    setBusy("analyse");
    setError(null);
    try {
      add({
        kind: "analysis",
        request: built.request,
        report: await api.odeAnalyze(built.request),
      });
    } catch (e) {
      fail(e, "The request failed. Try again.");
    } finally {
      setBusy(null);
    }
  }

  async function solveSymbolic(request: OdeInput | OdeSolveInput) {
    setBusy("symbolic");
    setError(null);
    try {
      add({ kind: "symbolic", request, report: await api.odeSolve(request) });
    } catch (e) {
      fail(e, "The request failed. Try again.");
    } finally {
      setBusy(null);
    }
  }

  /** Throws, so that the numerical form can show the message next to its fields. */
  async function solveNumeric(request: OdeNumericInput) {
    try {
      add({ kind: "numeric", request, report: await api.odeNumeric(request) });
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) onUnauthorized();
      throw e;
    }
  }

  function action(run: Extract<Run, { kind: "analysis" }>, m: MethodSuggestion) {
    const open = (form: FormName, label: string) => (
      <button
        type="button"
        className={ACTION}
        aria-expanded={formFor?.key === run.key && formFor.form === form}
        onClick={() =>
          setFormFor((f) => (f?.key === run.key && f.form === form ? null : { key: run.key, form }))
        }
      >
        {label}
      </button>
    );
    switch (m.id) {
      case "symbolic_closed_form":
        return (
          <button
            type="button"
            className={ACTION}
            disabled={busy !== null}
            onClick={() => void solveSymbolic(run.request)}
          >
            {busy === "symbolic" ? "Solving and checking…" : "Find a closed form"}
          </button>
        );
      case "numerical_ivp":
        return open("ivp", "Solve numerically…");
      case "numerical_bvp":
        return open("bvp", "Solve the boundary-value problem…");
      case "equilibrium_analysis":
      case "phase_plane":
        return open("qualitative", "Equilibria and stability…");
      case "power_series":
        return open("series", "Build a series…");
      case "regular_perturbation":
      case "poincare_lindstedt":
        return open("perturbation", "Expand in a small parameter…");
      default:
        return null;
    }
  }

  /** Throws, so that the form can show the message next to its fields. */
  async function solveExpansion(request: OdeSolveInput) {
    try {
      add({ kind: "symbolic", request, report: await api.odeSolve(request) });
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) onUnauthorized();
      throw e;
    }
  }

  /** Throws, so that the form can show the message next to its fields. */
  async function solveQualitative(request: OdeAnalyzeInput) {
    try {
      add({ kind: "analysis", request, report: await api.odeAnalyze(request) });
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) onUnauthorized();
      throw e;
    }
  }

  /** Verified closed forms of this same equation that a numerical run may be compared with. */
  function compareOptions(run: Extract<Run, { kind: "analysis" }>) {
    const same = (other: OdeInput) =>
      JSON.stringify([other.equations, other.format]) ===
      JSON.stringify([run.request.equations, run.request.format]);
    const unknowns = run.report.interpretation?.unknowns.length ?? 0;
    return runs.flatMap((r) =>
      r.kind === "symbolic" && same(r.request)
        ? r.report.solutions
            .filter(
              (item) =>
                item.kind === "explicit" &&
                item.constants.length === 0 &&
                item.rhs_text.length === unknowns,
            )
            .map((item) => ({
              label: `closed form (result ${r.key}): ${item.rhs_text.join(", ").slice(0, 60)}`,
              rhs: item.rhs_text,
            }))
        : [],
    );
  }

  /** Numerical trajectories of two states for the same equation, to draw over its direction field. */
  function trajectoriesFor(run: Extract<Run, { kind: "analysis" }>) {
    return runs.flatMap((r) =>
      r.kind === "numeric" &&
      r.report.status === "ok" &&
      r.report.series.length === 2 &&
      JSON.stringify(r.request.equations) === JSON.stringify(run.request.equations)
        ? [
            {
              label: `run ${r.key}`,
              x: r.report.series[0]?.values ?? [],
              y: r.report.series[1]?.values ?? [],
            },
          ]
        : [],
    );
  }

  function formBody(run: Extract<Run, { kind: "analysis" }>) {
    const meaning = run.report.interpretation;
    if (formFor?.key !== run.key || !meaning || run.report.status !== "ok") return null;
    const body =
      formFor.form === "qualitative" ? (
        <OdeQualitativeForm base={run.request} meaning={meaning} onRun={solveQualitative} />
      ) : formFor.form === "ivp" ? (
        <OdeNumericForm
          base={run.request}
          meaning={meaning}
          onRun={solveNumeric}
          compareOptions={compareOptions(run)}
        />
      ) : formFor.form === "bvp" ? (
        <OdeBoundaryForm base={run.request} meaning={meaning} onRun={solveNumeric} />
      ) : (
        <OdeSolveForm
          key={formFor.form}
          base={run.request}
          meaning={meaning}
          method={formFor.form}
          onRun={solveExpansion}
        />
      );
    return <div className="mt-4 rounded border border-rule p-3">{body}</div>;
  }

  return (
    <div className="min-w-0 space-y-4" data-testid="ode-workbench">
      <section
        aria-labelledby={`${uid}-heading`}
        className="space-y-4 rounded-lg border border-rule bg-raised p-4"
      >
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 id={`${uid}-heading`} className="font-serif text-xl">
            Differential equations
          </h2>
          <label className="flex items-center gap-2 text-xs text-muted">
            Start from
            <select
              aria-label="Start from an example"
              className={FIELD}
              value=""
              onChange={(e) => {
                const chosen = ODE_EXAMPLES.find((x) => x.id === e.target.value);
                if (chosen) {
                  setForm(chosen.form);
                  setRuns([]);
                  setError(null);
                  setFormFor(null);
                }
              }}
            >
              <option value="">an example…</option>
              {ODE_EXAMPLES.map((x) => (
                <option key={x.id} value={x.id}>
                  {x.title}
                </option>
              ))}
            </select>
          </label>
        </div>
        <p className="max-w-prose text-sm text-muted">
          Write one equation per line as <code>left = right</code>. Derivatives:{" "}
          <code>y&apos;</code>, <code>y&apos;&apos;</code>, <code>\dot y</code>,{" "}
          <code>\frac{"{d^2 y}{d t^2}"}</code>, <code>y^{"{(3)}"}</code>. Write a product after an
          unknown as <code>y \cdot (1 - y)</code>: <code>y(1 - y)</code> would be read as a function
          call. Minerva never chooses the independent variable, the initial values, the parameters
          or the interval for you.
        </p>

        <div className="space-y-2">
          <div className="flex flex-wrap items-center gap-3">
            <label className="text-xs uppercase tracking-wide text-muted" htmlFor={`${uid}-eq`}>
              Equations
            </label>
            <select
              aria-label="Input syntax"
              className={FIELD}
              value={form.format}
              onChange={(e) =>
                setForm((f) => ({ ...f, format: e.target.value as "latex" | "text" }))
              }
            >
              <option value="latex">LaTeX</option>
              <option value="text">Plain text</option>
            </select>
          </div>
          <textarea
            id={`${uid}-eq`}
            rows={3}
            className={`${FIELD} block w-full`}
            placeholder={form.format === "latex" ? "y'' + 0.2 y' + y + y^3 = 0" : "y'' + y = 0"}
            value={form.equations}
            onChange={(e) => setForm((f) => ({ ...f, equations: e.target.value }))}
          />
        </div>

        <div className="grid gap-3 sm:grid-cols-[12rem_1fr]">
          <label className="text-xs text-muted">
            Independent variable
            <input
              className={`${FIELD} mt-0.5 block w-full`}
              placeholder="t"
              value={form.independent}
              onChange={(e) => setForm((f) => ({ ...f, independent: e.target.value }))}
            />
          </label>
          <label className="text-xs text-muted">
            Conditions <span className="font-normal">(optional; one per line, like y(0) = 1)</span>
            <textarea
              rows={2}
              className={`${FIELD} mt-0.5 block w-full`}
              value={form.conditions}
              onChange={(e) => setForm((f) => ({ ...f, conditions: e.target.value }))}
            />
          </label>
        </div>

        <AssumptionsEditor
          value={form.assumptions}
          onChange={(assumptions) => setForm((f) => ({ ...f, assumptions }))}
        />

        {error && (
          <p role="alert" className="text-sm text-bad">
            {error}
          </p>
        )}
        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            disabled={busy !== null}
            onClick={() => void analyse()}
            className="rounded bg-accent px-4 py-2 text-sm font-medium text-accent-ink disabled:opacity-60"
          >
            {busy === "analyse" ? "Reading…" : "Analyse"}
          </button>
          {runs.length > 0 && (
            <button
              type="button"
              className="text-xs text-muted underline"
              onClick={() => {
                setRuns([]);
                setFormFor(null);
              }}
            >
              Clear the results on screen
            </button>
          )}
        </div>
      </section>

      {runs.map((run) => (
        <div key={run.key} data-run={run.kind}>
          {run.kind === "analysis" ? (
            <OdeAnalysisPanel
              report={run.report}
              actions={(m) => action(run, m)}
              trajectories={trajectoriesFor(run)}
            >
              {formBody(run)}
            </OdeAnalysisPanel>
          ) : run.kind === "symbolic" ? (
            <OdeSolutionPanel report={run.report} onUse={onUseExpression} />
          ) : (
            <OdeNumericPanel
              report={run.report}
              independent={run.request.independent ?? undefined}
            />
          )}
        </div>
      ))}
    </div>
  );
}
