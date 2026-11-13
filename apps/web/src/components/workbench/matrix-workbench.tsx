"use client";

import { useEffect, useId, useState } from "react";
import { ApiError, api } from "@/lib/api";
import {
  ACTIONS,
  EXAMPLES,
  MAX_MATRICES,
  blankMatrix,
  buildMatrixRequest,
  resize,
  setKind,
  type MatrixDeclForm,
  type MatrixForm,
  blankForm,
} from "@/lib/matrix-form";
import type { MatrixCalculusResult } from "@/lib/types";
import { AssumptionsEditor } from "../assumptions-editor";
import { MatrixPanel } from "../matrix-panel";

const FIELD =
  "min-w-0 rounded border border-rule bg-raised px-2 py-1.5 font-mono text-sm text-ink disabled:opacity-50";

interface Run {
  key: number;
  result: MatrixCalculusResult;
}

/**
 * Declare matrices once (a size, or explicit entries), then work on an expression in them. The
 * declarations stay in the form between runs and each saved run restores them. Every result is
 * verified by the server by the checks of its own method; no language model is involved, and the
 * server never assumes that matrices commute or are invertible: only what is declared below counts.
 */
export function MatrixWorkbench({
  restore,
  onUnauthorized,
  onSaved,
}: {
  /** A saved problem to load into the form (the nonce lets the same one be loaded again). */
  restore: { form: MatrixForm; nonce: number } | null;
  onUnauthorized: () => void;
  onSaved: () => void;
}) {
  const [form, setForm] = useState<MatrixForm>(() => blankForm());
  const [runs, setRuns] = useState<Run[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const uid = useId();
  const needsVariable = form.action !== "simplify" && form.action !== "evaluate";

  useEffect(() => {
    if (!restore) return;
    // eslint-disable-next-line react-hooks/set-state-in-effect -- loading a saved problem into the form
    setForm(restore.form);
    setRuns([]);
    setError(null);
  }, [restore]);

  async function compute() {
    const built = buildMatrixRequest(form);
    if (!built.ok) {
      setError(built.error);
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const result = await api.matrix(
        { input: built.input, assumptions: built.assumptions, format: built.format },
        built.request,
      );
      setRuns((r) => [...r, { key: (r.at(-1)?.key ?? 0) + 1, result }]);
      onSaved();
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) onUnauthorized();
      else setError(e instanceof ApiError ? e.message : "The request failed. Try again.");
    } finally {
      setBusy(false);
    }
  }

  function update(i: number, next: MatrixDeclForm) {
    setForm((f) => ({ ...f, matrices: f.matrices.map((m, k) => (k === i ? next : m)) }));
  }

  return (
    <div className="min-w-0 space-y-4" data-testid="matrix-workbench">
      <section
        aria-labelledby={`${uid}-heading`}
        className="space-y-4 rounded-lg border border-rule bg-raised p-4"
      >
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 id={`${uid}-heading`} className="font-serif text-xl">
            Matrices
          </h2>
          <label className="flex items-center gap-2 text-xs text-muted">
            Start from
            <select
              aria-label="Start from an example"
              className={FIELD}
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
        <p className="max-w-prose text-sm text-muted">
          Matrices are never assumed to commute or to be invertible: a step that needs either is
          reported as blocked until you declare it below. Write exponentials as \exp(tA) (a bare e
          is a symbol) and a time-dependent matrix as A(t) after naming its variable.
        </p>

        <div className="space-y-2">
          <div className="flex flex-wrap items-center gap-3">
            <label className="text-xs uppercase tracking-wide text-muted" htmlFor={`${uid}-expr`}>
              Expression
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
          <input
            id={`${uid}-expr`}
            className={`${FIELD} block w-full`}
            placeholder={form.format === "latex" ? "\\exp(A)\\exp(B)" : "exp(A)*exp(B)"}
            value={form.input}
            onChange={(e) => setForm((f) => ({ ...f, input: e.target.value }))}
          />
        </div>

        <fieldset className="space-y-3">
          <legend className="text-xs uppercase tracking-wide text-muted">Declared matrices</legend>
          {form.matrices.map((m, i) => (
            <div key={i} className="space-y-2 rounded border border-rule p-3">
              <div className="flex flex-wrap items-end gap-3">
                <label className="text-xs text-muted">
                  Name
                  <input
                    aria-label={`Matrix ${i + 1} name`}
                    className={`${FIELD} mt-0.5 block w-20`}
                    value={m.name}
                    onChange={(e) => update(i, { ...m, name: e.target.value })}
                  />
                </label>
                <label className="text-xs text-muted">
                  Given as
                  <select
                    aria-label={`Matrix ${i + 1} kind`}
                    className={`${FIELD} mt-0.5 block`}
                    value={m.kind}
                    onChange={(e) =>
                      update(i, setKind(m, e.target.value as "abstract" | "explicit"))
                    }
                  >
                    <option value="abstract">a size</option>
                    <option value="explicit">its entries</option>
                  </select>
                </label>
                <label className="text-xs text-muted">
                  Rows
                  <input
                    type="number"
                    min={1}
                    aria-label={`Matrix ${i + 1} rows`}
                    className={`${FIELD} mt-0.5 block w-16`}
                    value={m.rows}
                    onChange={(e) => update(i, resize(m, Number(e.target.value), m.cols))}
                  />
                </label>
                <label className="text-xs text-muted">
                  Columns
                  <input
                    type="number"
                    min={1}
                    aria-label={`Matrix ${i + 1} columns`}
                    className={`${FIELD} mt-0.5 block w-16`}
                    value={m.cols}
                    onChange={(e) => update(i, resize(m, m.rows, Number(e.target.value)))}
                  />
                </label>
                <label className="text-xs text-muted">
                  Depends on (optional)
                  <input
                    aria-label={`Matrix ${i + 1} depends on`}
                    placeholder="t"
                    className={`${FIELD} mt-0.5 block w-20`}
                    value={m.dependsOn}
                    onChange={(e) => update(i, { ...m, dependsOn: e.target.value })}
                  />
                </label>
                {form.matrices.length > 1 && (
                  <button
                    type="button"
                    className="text-sm text-muted hover:underline"
                    onClick={() =>
                      setForm((f) => ({ ...f, matrices: f.matrices.filter((_, k) => k !== i) }))
                    }
                  >
                    Remove
                  </button>
                )}
              </div>
              {m.kind === "explicit" && (
                <div className="overflow-x-auto pb-1">
                  <div
                    className="grid gap-2"
                    style={{ gridTemplateColumns: `repeat(${m.cols}, minmax(6rem, 1fr))` }}
                  >
                    {m.entries.map((row, r) =>
                      row.map((cell, c) => (
                        <input
                          key={`${r}-${c}`}
                          aria-label={`${m.name || "Matrix"} entry ${r + 1},${c + 1}`}
                          className={FIELD}
                          value={cell}
                          onChange={(e) =>
                            update(i, {
                              ...m,
                              entries: m.entries.map((rr, ri) =>
                                ri === r
                                  ? rr.map((cc, ci) => (ci === c ? e.target.value : cc))
                                  : rr,
                              ),
                            })
                          }
                        />
                      )),
                    )}
                  </div>
                </div>
              )}
            </div>
          ))}
          {form.matrices.length < MAX_MATRICES && (
            <button
              type="button"
              className="text-sm text-accent hover:underline"
              onClick={() =>
                setForm((f) => ({
                  ...f,
                  matrices: [
                    ...f.matrices,
                    blankMatrix(
                      "ABCDEFGH".split("").find((n) => !f.matrices.some((m) => m.name === n)) ??
                        `M${f.matrices.length + 1}`,
                    ),
                  ],
                }))
              }
            >
              Add a matrix
            </button>
          )}
        </fieldset>

        <fieldset className="grid gap-3 sm:grid-cols-3">
          <legend className="text-xs uppercase tracking-wide text-muted">
            What you state about them (only you can)
          </legend>
          <label className="text-xs text-muted">
            Commuting pairs
            <input
              aria-label="Commuting pairs"
              placeholder="A,B; B,C"
              className={`${FIELD} mt-0.5 block w-full`}
              value={form.commuting}
              onChange={(e) => setForm((f) => ({ ...f, commuting: e.target.value }))}
            />
          </label>
          <label className="text-xs text-muted">
            Invertible
            <input
              aria-label="Invertible matrices"
              placeholder="A, B"
              className={`${FIELD} mt-0.5 block w-full`}
              value={form.invertible}
              onChange={(e) => setForm((f) => ({ ...f, invertible: e.target.value }))}
            />
          </label>
          <label className="text-xs text-muted">
            Commute with themselves at all times
            <input
              aria-label="Self-commuting matrices"
              placeholder="A"
              className={`${FIELD} mt-0.5 block w-full`}
              value={form.selfCommuting}
              onChange={(e) => setForm((f) => ({ ...f, selfCommuting: e.target.value }))}
            />
          </label>
        </fieldset>

        <AssumptionsEditor
          value={form.assumptions}
          onChange={(assumptions) => setForm((f) => ({ ...f, assumptions }))}
        />

        <fieldset className="space-y-2">
          <legend className="text-xs uppercase tracking-wide text-muted">Do</legend>
          <div className="flex flex-wrap gap-x-4 gap-y-1">
            {ACTIONS.map((a) => (
              <label key={a.id} className="flex items-center gap-1.5 text-sm" title={a.hint}>
                <input
                  type="radio"
                  name={`${uid}-action`}
                  checked={form.action === a.id}
                  onChange={() => setForm((f) => ({ ...f, action: a.id }))}
                />
                {a.label}
              </label>
            ))}
          </div>
          {needsVariable && (
            <div className="flex flex-wrap gap-3">
              <label className="text-xs text-muted">
                Variable
                <input
                  aria-label="Variable"
                  className={`${FIELD} mt-0.5 block w-24`}
                  value={form.variable}
                  onChange={(e) => setForm((f) => ({ ...f, variable: e.target.value }))}
                />
              </label>
              <label className="text-xs text-muted">
                {form.action === "evolve" ? "Start (optional)" : "Lower bound (optional)"}
                <input
                  aria-label="Lower bound"
                  className={`${FIELD} mt-0.5 block w-28`}
                  value={form.lower}
                  onChange={(e) => setForm((f) => ({ ...f, lower: e.target.value }))}
                />
              </label>
              <label className="text-xs text-muted">
                {form.action === "evolve"
                  ? "End of the span (you choose)"
                  : "Upper bound (optional)"}
                <input
                  aria-label="Upper bound"
                  className={`${FIELD} mt-0.5 block w-28`}
                  value={form.upper}
                  onChange={(e) => setForm((f) => ({ ...f, upper: e.target.value }))}
                />
              </label>
            </div>
          )}
        </fieldset>

        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={() => void compute()}
            disabled={busy}
            className="rounded bg-accent px-4 py-2 font-medium text-accent-ink disabled:opacity-50"
          >
            {busy ? "Computing and checking…" : "Compute"}
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
        <MatrixPanel key={run.key} result={run.result} headingId={`matrix-heading-${run.key}`} />
      ))}
    </div>
  );
}
