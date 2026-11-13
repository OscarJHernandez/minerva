"use client";

import { useEffect, useState } from "react";
import { ApiError, api } from "@/lib/api";
import {
  EXAMPLES,
  MAX_DIMENSION,
  QUANTITY_OPTIONS,
  blankForm,
  buildGeometryRequest,
  restoreForm,
  setDimension,
  setMetricCell,
  setRank,
  type GeometryForm,
} from "@/lib/geometry-form";
import type { GeometryQuantity, GeometryRequest, GeometryResult } from "@/lib/types";
import { AssumptionsEditor } from "../assumptions-editor";
import { GeometryPanel } from "../geometry-panel";

const FIELD =
  "min-w-0 rounded border border-rule bg-raised px-2 py-1.5 font-mono text-sm text-ink disabled:opacity-50";

interface Run {
  key: number;
  request: GeometryRequest;
  result: GeometryResult;
}

/**
 * Define a geometric problem once (coordinates, metric, assumptions, functions) and ask for several
 * quantities of it. The metric stays in the form between runs, and each saved run restores it. Every
 * result is verified by the server; no language model is involved.
 */
export function GeometryWorkbench({
  restore,
  onUnauthorized,
  onSaved,
}: {
  /** A saved problem to load into the form (the nonce lets the same one be loaded again). */
  restore: { request: GeometryRequest; nonce: number } | null;
  onUnauthorized: () => void;
  onSaved: () => void;
}) {
  const [form, setForm] = useState<GeometryForm>(() => blankForm(2));
  const [runs, setRuns] = useState<Run[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const n = form.coordinates.length;
  const needsTensor = form.quantities.some((q) => QUANTITY_OPTIONS.find((o) => o.id === q)?.tensor);
  const needsIndex = form.quantities.some((q) => q === "raise_index" || q === "lower_index");

  useEffect(() => {
    if (!restore) return;
    // eslint-disable-next-line react-hooks/set-state-in-effect -- loading a saved problem into the form
    setForm(restoreForm(restore.request));
    setRuns([]);
    setError(null);
  }, [restore]);

  async function execute(request: GeometryRequest) {
    setBusy(true);
    setError(null);
    try {
      const result = await api.geometry(request);
      setRuns((r) => [...r, { key: (r.at(-1)?.key ?? 0) + 1, request, result }]);
      onSaved();
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) onUnauthorized();
      else setError(e instanceof ApiError ? e.message : "The request failed. Try again.");
    } finally {
      setBusy(false);
    }
  }

  function compute() {
    const built = buildGeometryRequest(form);
    if (!built.ok) {
      setError(built.error);
      return;
    }
    void execute(built.request);
  }

  function toggleQuantity(id: GeometryQuantity) {
    setForm((f) => ({
      ...f,
      quantities: f.quantities.includes(id)
        ? f.quantities.filter((q) => q !== id)
        : [...f.quantities, id],
    }));
  }

  return (
    <div className="min-w-0 space-y-4" data-testid="geometry-workbench">
      <section
        aria-labelledby="geometry-form-heading"
        className="space-y-4 rounded-lg border border-rule bg-raised p-4"
      >
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 id="geometry-form-heading" className="font-serif text-xl">
            Geometry
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
          Give the coordinates and the metric components g<sub>μν</sub> as LaTeX. Nothing is assumed
          about the signature; coordinates are real; other symbols are complex unless you declare
          them below. An empty off-diagonal entry means 0.
        </p>

        <div className="flex flex-wrap items-end gap-3">
          <label className="text-xs text-muted">
            Dimension
            <select
              aria-label="Dimension"
              className={`${FIELD} ml-2`}
              value={n}
              onChange={(e) => setForm((f) => setDimension(f, Number(e.target.value)))}
            >
              {Array.from({ length: MAX_DIMENSION }, (_, i) => i + 1).map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
          </label>
          {form.coordinates.map((name, i) => (
            <label key={i} className="text-xs text-muted">
              Coordinate {i + 1}
              <input
                aria-label={`Coordinate ${i + 1}`}
                className={`${FIELD} mt-0.5 block w-24`}
                value={name}
                onChange={(e) =>
                  setForm((f) => ({
                    ...f,
                    coordinates: f.coordinates.map((c, k) => (k === i ? e.target.value : c)),
                  }))
                }
              />
            </label>
          ))}
        </div>

        <fieldset>
          <legend className="text-xs uppercase tracking-wide text-muted">Metric g</legend>
          <div className="mt-1 overflow-x-auto pb-1">
            <div
              className="grid gap-2"
              style={{ gridTemplateColumns: `repeat(${n}, minmax(9rem, 1fr))` }}
            >
              {form.metric.map((row, i) =>
                row.map((cell, j) => (
                  <input
                    key={`${i}-${j}`}
                    aria-label={`Metric component ${i + 1},${j + 1}`}
                    placeholder={i === j ? "required" : "0"}
                    className={FIELD}
                    value={cell}
                    onChange={(e) => setForm((f) => setMetricCell(f, i, j, e.target.value))}
                  />
                )),
              )}
            </div>
          </div>
        </fieldset>

        <label className="block text-xs text-muted">
          Unspecified functions (optional), e.g. f(r), a(t)
          <input
            aria-label="Functions"
            className={`${FIELD} mt-0.5 block w-full`}
            value={form.functions}
            onChange={(e) => setForm((f) => ({ ...f, functions: e.target.value }))}
          />
        </label>

        <AssumptionsEditor
          value={form.assumptions}
          onChange={(assumptions) => setForm((f) => ({ ...f, assumptions }))}
        />

        <fieldset>
          <legend className="text-xs uppercase tracking-wide text-muted">Compute</legend>
          <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1">
            {QUANTITY_OPTIONS.map((o) => (
              <label key={o.id} className="flex items-center gap-1.5 text-sm">
                <input
                  type="checkbox"
                  checked={form.quantities.includes(o.id)}
                  onChange={() => toggleQuantity(o.id)}
                />
                {o.label}
              </label>
            ))}
          </div>
        </fieldset>

        {needsTensor && (
          <fieldset className="space-y-2 rounded border border-rule p-3">
            <legend className="px-1 text-xs uppercase tracking-wide text-muted">
              Tensor field
            </legend>
            <div className="flex flex-wrap items-center gap-3 text-sm">
              <label className="text-xs text-muted">
                Rank
                <select
                  aria-label="Tensor rank"
                  className={`${FIELD} ml-2`}
                  value={form.tensorVariance.length}
                  onChange={(e) => setForm((f) => setRank(f, Number(e.target.value)))}
                >
                  {[1, 2, 3].map((r) => (
                    <option key={r} value={r}>
                      {r}
                    </option>
                  ))}
                </select>
              </label>
              {form.tensorVariance.map((v, k) => (
                <label key={k} className="text-xs text-muted">
                  Index {k + 1}
                  <select
                    aria-label={`Index ${k + 1} position`}
                    className={`${FIELD} ml-2`}
                    value={v}
                    onChange={(e) =>
                      setForm((f) => ({
                        ...f,
                        tensorVariance: f.tensorVariance.map((x, i) =>
                          i === k ? (e.target.value as "up" | "down") : x,
                        ),
                      }))
                    }
                  >
                    <option value="up">up (V^μ)</option>
                    <option value="down">down (ω_μ)</option>
                  </select>
                </label>
              ))}
              {needsIndex && form.tensorVariance.length > 1 && (
                <label className="text-xs text-muted">
                  Raise or lower index
                  <select
                    aria-label="Index to raise or lower"
                    className={`${FIELD} ml-2`}
                    value={form.tensorIndex}
                    onChange={(e) =>
                      setForm((f) => ({ ...f, tensorIndex: Number(e.target.value) }))
                    }
                  >
                    {form.tensorVariance.map((_, k) => (
                      <option key={k} value={k}>
                        {k + 1}
                      </option>
                    ))}
                  </select>
                </label>
              )}
            </div>
            <div className="overflow-x-auto pb-1">
              <div
                className="grid gap-2"
                style={{ gridTemplateColumns: `repeat(${n}, minmax(8rem, 1fr))` }}
              >
                {form.tensorComponents.map((c, k) => (
                  <input
                    key={k}
                    aria-label={`Tensor component ${k + 1}`}
                    className={FIELD}
                    value={c}
                    onChange={(e) =>
                      setForm((f) => ({
                        ...f,
                        tensorComponents: f.tensorComponents.map((x, i) =>
                          i === k ? e.target.value : x,
                        ),
                      }))
                    }
                  />
                ))}
              </div>
            </div>
            <p className="text-xs text-muted">
              Components in row-major order (the first index varies slowest), as LaTeX.
            </p>
          </fieldset>
        )}

        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={compute}
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
        <GeometryPanel
          key={run.key}
          result={run.result}
          asked={runs.filter((r) => r.key > run.key).flatMap((r) => r.request.quantities)}
          busy={busy}
          onMore={(q) => void execute({ ...run.request, quantities: [q], tensor: null })}
        />
      ))}
    </div>
  );
}
