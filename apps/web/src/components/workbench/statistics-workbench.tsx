"use client";

import { useEffect, useId, useState } from "react";
import { ApiError, api } from "@/lib/api";
import {
  FAMILIES,
  PRIORS,
  STAT_ACTIONS,
  STAT_EXAMPLES,
  blankStatForm,
  buildStatRequest,
  familyInfo,
  paramNames,
  withFamily,
  type StatForm,
} from "@/lib/statistics-form";
import type { StatModelResult } from "@/lib/types";
import { AssumptionsEditor } from "../assumptions-editor";
import { StatisticsPanel } from "../statistics-panel";

const FIELD =
  "min-w-0 max-w-full rounded border border-rule bg-raised px-2 py-1.5 font-mono text-sm text-ink disabled:opacity-50";

interface Run {
  key: number;
  result: StatModelResult;
}

/**
 * Pick a family, say which parameter is unknown, and derive: moments, a log-likelihood, a score, a
 * maximum-likelihood estimator, Fisher information or a posterior. Independence is a box you tick
 * yourself and is off by default: with several observations and the box unticked, the server refuses
 * and the result says why. Nothing here calls a language model.
 */
export function StatisticsWorkbench({
  restore,
  onUnauthorized,
  onSaved,
}: {
  restore: { form: StatForm; nonce: number } | null;
  onUnauthorized: () => void;
  onSaved: () => void;
}) {
  const [form, setForm] = useState<StatForm>(() => blankStatForm());
  const [runs, setRuns] = useState<Run[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const uid = useId();
  const info = familyInfo(form.family);
  const names = paramNames(form.family, form.parametrization);
  const priorInfo = familyInfo(form.priorFamily);
  const priorNames = paramNames(form.priorFamily, form.priorParametrization);
  const usesData = ["log_likelihood", "score", "mle", "posterior"].includes(form.action);
  const several = form.sampleSize.trim() !== "" || form.data.trim() !== "";

  useEffect(() => {
    if (!restore) return;
    // eslint-disable-next-line react-hooks/set-state-in-effect -- loading a saved problem into the form
    setForm(restore.form);
    setRuns([]);
    setError(null);
  }, [restore]);

  async function compute() {
    const built = buildStatRequest(form);
    if (!built.ok) {
      setError(built.error);
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const result = await api.statistics(built.request);
      setRuns((r) => [...r, { key: (r.at(-1)?.key ?? 0) + 1, result }]);
      onSaved();
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) onUnauthorized();
      else setError(e instanceof ApiError ? e.message : "The request failed. Try again.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="min-w-0 space-y-4" data-testid="statistics-workbench">
      <section
        aria-labelledby={`${uid}-heading`}
        className="space-y-4 rounded-lg border border-rule bg-raised p-4"
      >
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 id={`${uid}-heading`} className="font-serif text-xl">
            Statistics
          </h2>
          <label className="flex min-w-0 max-w-full items-center gap-2 text-xs text-muted">
            Start from
            <select
              aria-label="Start from an example"
              className={FIELD}
              value=""
              onChange={(e) => {
                const chosen = STAT_EXAMPLES.find((x) => x.id === e.target.value);
                if (chosen) {
                  setForm(chosen.form);
                  setRuns([]);
                  setError(null);
                }
              }}
            >
              <option value="">an example…</option>
              {STAT_EXAMPLES.map((x) => (
                <option key={x.id} value={x.id}>
                  {x.label}
                </option>
              ))}
            </select>
          </label>
        </div>
        <p className="max-w-prose text-sm text-muted">
          Results are derived from the density of the family and checked by a different route (a
          direct numerical sum, quadrature). Independence is never assumed: tick it only if you
          state it. A prior is your assumption, and the result says so.
        </p>

        <fieldset className="min-w-0 space-y-2">
          <legend className="text-xs uppercase tracking-wide text-muted">Derive</legend>
          <div className="flex flex-wrap gap-x-4 gap-y-1">
            {STAT_ACTIONS.map((a) => (
              <label key={a.id} className="flex items-center gap-1.5 text-sm" title={a.hint}>
                <input
                  type="radio"
                  name={`${uid}-action`}
                  checked={form.action === a.id}
                  onChange={() =>
                    setForm((f) => ({
                      ...f,
                      action: a.id,
                      // moments need every parameter's value: nothing is left unknown
                      params:
                        a.id === "moments"
                          ? Object.fromEntries(
                              Object.entries(f.params).map(([n, p]) => [
                                n,
                                { ...p, unknown: false },
                              ]),
                            )
                          : f.params,
                    }))
                  }
                />
                {a.label}
              </label>
            ))}
          </div>
        </fieldset>

        <fieldset className="min-w-0 space-y-3">
          <legend className="text-xs uppercase tracking-wide text-muted">Model</legend>
          <div className="flex flex-wrap items-end gap-3">
            <label className="min-w-0 max-w-full text-xs text-muted">
              Family
              <select
                aria-label="Family"
                className={`${FIELD} mt-0.5 block`}
                value={form.family}
                onChange={(e) => setForm((f) => withFamily(f, e.target.value))}
              >
                {FAMILIES.map((x) => (
                  <option key={x.id} value={x.id}>
                    {x.title}
                  </option>
                ))}
              </select>
            </label>
            {Object.keys(info.parametrizations).length > 1 && (
              <label className="min-w-0 max-w-full text-xs text-muted">
                Parametrisation
                <select
                  aria-label="Parametrisation"
                  className={`${FIELD} mt-0.5 block`}
                  value={form.parametrization}
                  onChange={(e) => setForm((f) => withFamily(f, f.family, e.target.value))}
                >
                  <option value="sd">mean and standard deviation (mu, sigma)</option>
                  <option value="variance">mean and variance (mu, v)</option>
                </select>
              </label>
            )}
          </div>
          <div className="space-y-2">
            {names.map((n) => (
              <div key={n} className="flex flex-wrap items-center gap-3">
                <span className="w-16 font-mono text-sm">{n}</span>
                <input
                  aria-label={`Value of ${n}`}
                  className={`${FIELD} w-40`}
                  placeholder={
                    info.defaults?.[n] ? `${info.defaults[n]} (standard)` : "value or name"
                  }
                  disabled={form.params[n]?.unknown}
                  value={form.params[n]?.value ?? ""}
                  onChange={(e) =>
                    setForm((f) => ({
                      ...f,
                      params: { ...f.params, [n]: { ...f.params[n]!, value: e.target.value } },
                    }))
                  }
                />
                {form.action !== "moments" && (
                  <label className="flex items-center gap-1.5 text-xs text-muted">
                    <input
                      type="checkbox"
                      aria-label={`${n} is unknown`}
                      checked={form.params[n]?.unknown ?? false}
                      onChange={(e) =>
                        setForm((f) => ({
                          ...f,
                          params: {
                            ...f.params,
                            [n]: { ...f.params[n]!, unknown: e.target.checked },
                          },
                        }))
                      }
                    />
                    unknown
                  </label>
                )}
              </div>
            ))}
          </div>
        </fieldset>

        {(usesData || form.action === "fisher_information") && (
          <fieldset className="min-w-0 space-y-3">
            <legend className="text-xs uppercase tracking-wide text-muted">Observations</legend>
            <div className="flex flex-wrap items-end gap-3">
              {usesData && (
                <label className="text-xs text-muted">
                  Name of the data
                  <input
                    aria-label="Name of the data"
                    className={`${FIELD} mt-0.5 block w-20`}
                    value={form.observation}
                    onChange={(e) => setForm((f) => ({ ...f, observation: e.target.value }))}
                  />
                </label>
              )}
              <label className="text-xs text-muted">
                Number of observations (empty: one)
                <input
                  aria-label="Number of observations"
                  className={`${FIELD} mt-0.5 block w-28`}
                  placeholder="n"
                  value={form.sampleSize}
                  onChange={(e) => setForm((f) => ({ ...f, sampleSize: e.target.value }))}
                />
              </label>
              {usesData && (
                <label className="min-w-40 flex-1 text-xs text-muted">
                  Concrete data (optional)
                  <input
                    aria-label="Concrete data"
                    className={`${FIELD} mt-0.5 block w-full`}
                    placeholder="1, 0, 1, 1"
                    value={form.data}
                    onChange={(e) => setForm((f) => ({ ...f, data: e.target.value }))}
                  />
                </label>
              )}
            </div>
            <label className="flex items-start gap-2 text-sm">
              <input
                type="checkbox"
                className="mt-1"
                checked={form.iid}
                disabled={!several}
                onChange={(e) => setForm((f) => ({ ...f, iid: e.target.checked }))}
              />
              <span>
                The observations are independent and identically distributed
                <span className="block text-xs text-muted">
                  Your statement, not checked by Minerva. Without it, several observations are
                  refused.
                </span>
              </span>
            </label>
          </fieldset>
        )}

        {form.action === "posterior" && (
          <fieldset className="min-w-0 space-y-3">
            <legend className="text-xs uppercase tracking-wide text-muted">
              Prior (your assumption)
            </legend>
            <div className="flex flex-wrap items-end gap-3">
              <label className="text-xs text-muted">
                Prior family
                <select
                  aria-label="Prior family"
                  className={`${FIELD} mt-0.5 block`}
                  value={form.priorFamily}
                  onChange={(e) =>
                    setForm((f) => ({
                      ...f,
                      priorFamily: e.target.value,
                      priorParametrization: e.target.value === "normal" ? "variance" : "default",
                      priorParams: {},
                    }))
                  }
                >
                  {PRIORS.map((id) => (
                    <option key={id} value={id}>
                      {familyInfo(id).title}
                    </option>
                  ))}
                </select>
              </label>
              {priorNames.map((n) => (
                <label key={n} className="text-xs text-muted">
                  {n}
                  <input
                    aria-label={`Prior ${n}`}
                    className={`${FIELD} mt-0.5 block w-28`}
                    value={form.priorParams[n] ?? ""}
                    onChange={(e) =>
                      setForm((f) => ({
                        ...f,
                        priorParams: { ...f.priorParams, [n]: e.target.value },
                      }))
                    }
                  />
                </label>
              ))}
            </div>
            {priorInfo.id === "normal" && (
              <p className="text-xs text-muted">The prior variance is v; its mean is mu.</p>
            )}
          </fieldset>
        )}

        <AssumptionsEditor
          value={form.assumptions}
          onChange={(assumptions) => setForm((f) => ({ ...f, assumptions }))}
        />

        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={() => void compute()}
            disabled={busy}
            className="rounded bg-accent px-4 py-2 font-medium text-accent-ink disabled:opacity-50"
          >
            {busy ? "Deriving and checking…" : "Derive"}
          </button>
          {runs.length > 0 && (
            <button
              type="button"
              onClick={() => setRuns([])}
              className="text-sm text-muted hover:underline"
            >
              Clear results
            </button>
          )}
        </div>
        {error && (
          <p role="alert" className="rounded bg-bad-soft px-3 py-2 text-sm text-bad">
            {error}
          </p>
        )}
      </section>

      {runs.map((run) => (
        <StatisticsPanel
          key={run.key}
          kind="statistics"
          result={run.result}
          headingId={`statistics-heading-${run.key}`}
        />
      ))}
    </div>
  );
}
