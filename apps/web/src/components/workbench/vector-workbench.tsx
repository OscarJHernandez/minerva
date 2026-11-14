"use client";

import { useEffect, useId, useState } from "react";
import { ApiError, api } from "@/lib/api";
import type { VectorRequest, VectorResult } from "@/lib/types";
import {
  ACTIONS,
  MAX_FIELDS,
  PREMISE_OPTIONS,
  TEMPLATES,
  blankField,
  blankForm,
  buildVectorRequest,
  type VectorFieldForm,
  type VectorForm,
} from "@/lib/vector-form";
import { AssumptionsEditor } from "../assumptions-editor";
import { VectorPanel } from "../vector-panel";

const FIELD =
  "min-w-0 rounded border border-rule bg-raised px-2 py-1.5 font-mono text-sm text-ink disabled:opacity-50";

interface Run {
  key: number;
  request: VectorRequest;
  result: VectorResult;
}

/**
 * Declare scalar and vector fields once, then take one step at a time: evaluate an operator, find or apply a
 * handbook identity at a part you name, substitute a field equation you adopted, or run a named
 * electromagnetic derivation. Each result is verified by the server and saved; a result can become the
 * expression of the next step. A field equation is a premise only because you adopted it (nothing is
 * adopted for you), and a result that used one is shown as holding given it. No language model is involved.
 */
export function VectorWorkbench({
  restore,
  onUnauthorized,
  onSaved,
}: {
  /** A saved problem to load into the form (the nonce lets the same one be loaded again). */
  restore: { form: VectorForm; nonce: number } | null;
  onUnauthorized: () => void;
  onSaved: () => void;
}) {
  const [form, setForm] = useState<VectorForm>(() => blankForm());
  const [runs, setRuns] = useState<Run[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const uid = useId();
  const em = form.action === "em_derive";
  const template = TEMPLATES.find((t) => t.id === form.template);

  useEffect(() => {
    if (!restore) return;
    // eslint-disable-next-line react-hooks/set-state-in-effect -- loading a saved problem into the form
    setForm(restore.form);
    setRuns([]);
    setError(null);
  }, [restore]);

  async function compute() {
    const built = buildVectorRequest(form);
    if (!built.ok) {
      setError(built.error);
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const result = await api.vector(built.request, { save: true });
      setRuns((r) => [...r, { key: (r.at(-1)?.key ?? 0) + 1, request: built.request, result }]);
      onSaved();
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) onUnauthorized();
      else setError(e instanceof ApiError ? e.message : "The request failed. Try again.");
    } finally {
      setBusy(false);
    }
  }

  function updateField(i: number, next: VectorFieldForm) {
    setForm((f) => ({ ...f, fields: f.fields.map((x, k) => (k === i ? next : x)) }));
  }

  function toggleAdopted(id: string) {
    setForm((f) => ({
      ...f,
      premiseIds: f.premiseIds.includes(id)
        ? f.premiseIds.filter((p) => p !== id)
        : [...f.premiseIds, id],
    }));
  }

  /** The verified result becomes the expression of the next step (LaTeX, as the server printed it). */
  function continueFrom(result: VectorResult) {
    if (!result.result_latex) return;
    setForm((f) => ({
      ...f,
      expression: result.result_latex ?? "",
      format: "latex",
      action: f.action === "em_derive" ? "evaluate" : f.action,
    }));
    setError(null);
  }

  return (
    <div className="min-w-0 space-y-4" data-testid="vector-workbench">
      <section
        aria-labelledby={`${uid}-heading`}
        className="space-y-4 rounded-lg border border-rule bg-raised p-4"
      >
        <h2 id={`${uid}-heading`} className="font-serif text-xl">
          Vector calculus
        </h2>
        <p className="max-w-prose text-sm text-muted">
          Nothing is assumed smooth: a step that needs equal mixed partial derivatives is blocked
          until you declare the field twice differentiable. Field equations are premises only if you
          adopt them, and Minerva never checks that they are true.
        </p>

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
        </fieldset>

        {em ? (
          <div className="space-y-3">
            <label className="block text-xs text-muted">
              Derivation
              <select
                aria-label="Derivation"
                className={`${FIELD} mt-0.5 block w-full max-w-full`}
                value={form.template}
                onChange={(e) => setForm((f) => ({ ...f, template: e.target.value }))}
              >
                {TEMPLATES.map((t) => (
                  <option key={t.id} value={t.id}>
                    {t.label}
                  </option>
                ))}
              </select>
            </label>
            {template && template.needs.length > 0 && (
              <p className="text-sm text-muted" data-testid="needs">
                Needs these field equations as premises (you adopt them below):{" "}
                {template.needs.join("; ")}
              </p>
            )}
            <label className="block text-xs text-muted">
              Smoothness of the fields (only you can declare it)
              <select
                aria-label="Smoothness"
                className={`${FIELD} mt-0.5 block w-full max-w-full`}
                value={form.emSmoothness}
                onChange={(e) =>
                  setForm((f) => ({
                    ...f,
                    emSmoothness: e.target.value as VectorForm["emSmoothness"],
                  }))
                }
              >
                <option value="">not declared</option>
                <option value="c2">twice continuously differentiable</option>
                <option value="smooth">smooth</option>
              </select>
            </label>
          </div>
        ) : (
          <>
            <div className="flex flex-wrap items-end gap-3">
              <label className="text-xs text-muted">
                Coordinates
                <select
                  aria-label="Coordinate system"
                  className={`${FIELD} mt-0.5 block`}
                  value={form.coordinates}
                  onChange={(e) =>
                    setForm((f) => ({
                      ...f,
                      coordinates: e.target.value as VectorForm["coordinates"],
                    }))
                  }
                >
                  <option value="cartesian">Cartesian (x, y, z)</option>
                  <option value="cylindrical">cylindrical (rho, phi, z)</option>
                  <option value="spherical">spherical (r, theta, phi)</option>
                </select>
              </label>
              <label className="text-xs text-muted">
                Input syntax
                <select
                  aria-label="Input syntax"
                  className={`${FIELD} mt-0.5 block`}
                  value={form.format}
                  onChange={(e) =>
                    setForm((f) => ({ ...f, format: e.target.value as "latex" | "text" }))
                  }
                >
                  <option value="text">Plain text</option>
                  <option value="latex">LaTeX</option>
                </select>
              </label>
            </div>
            <div className="space-y-1">
              <label className="text-xs uppercase tracking-wide text-muted" htmlFor={`${uid}-expr`}>
                Expression
              </label>
              <input
                id={`${uid}-expr`}
                className={`${FIELD} block w-full`}
                placeholder={form.format === "text" ? "curl(f*A)" : "\\nabla\\times(f\\mathbf{A})"}
                value={form.expression}
                onChange={(e) => setForm((f) => ({ ...f, expression: e.target.value }))}
              />
            </div>

            <fieldset className="space-y-3">
              <legend className="text-xs uppercase tracking-wide text-muted">
                Declared fields
              </legend>
              {form.fields.map((fld, i) => (
                <div
                  key={i}
                  className="flex flex-wrap items-end gap-3 rounded border border-rule p-3"
                >
                  <label className="text-xs text-muted">
                    Name
                    <input
                      aria-label={`Field ${i + 1} name`}
                      className={`${FIELD} mt-0.5 block w-20`}
                      value={fld.name}
                      onChange={(e) => updateField(i, { ...fld, name: e.target.value })}
                    />
                  </label>
                  <label className="text-xs text-muted">
                    Kind
                    <select
                      aria-label={`Field ${i + 1} kind`}
                      className={`${FIELD} mt-0.5 block`}
                      value={fld.kind}
                      onChange={(e) =>
                        updateField(i, { ...fld, kind: e.target.value as "scalar" | "vector" })
                      }
                    >
                      <option value="vector">vector</option>
                      <option value="scalar">scalar</option>
                    </select>
                  </label>
                  <label className="text-xs text-muted">
                    Components (optional)
                    <input
                      aria-label={`Field ${i + 1} components`}
                      placeholder={fld.kind === "vector" ? "x*y, 0, z" : "x^2+y"}
                      className={`${FIELD} mt-0.5 block w-44`}
                      value={fld.components}
                      onChange={(e) => updateField(i, { ...fld, components: e.target.value })}
                    />
                  </label>
                  <label className="text-xs text-muted">
                    Smoothness (yours to state)
                    <select
                      aria-label={`Field ${i + 1} smoothness`}
                      className={`${FIELD} mt-0.5 block`}
                      value={fld.smoothness}
                      onChange={(e) =>
                        updateField(i, {
                          ...fld,
                          smoothness: e.target.value as VectorFieldForm["smoothness"],
                        })
                      }
                    >
                      <option value="">not declared</option>
                      <option value="c1">once differentiable</option>
                      <option value="c2">twice differentiable</option>
                      <option value="smooth">smooth</option>
                    </select>
                  </label>
                  {form.fields.length > 1 && (
                    <button
                      type="button"
                      className="text-sm text-muted hover:underline"
                      onClick={() =>
                        setForm((f) => ({ ...f, fields: f.fields.filter((_, k) => k !== i) }))
                      }
                    >
                      Remove
                    </button>
                  )}
                </div>
              ))}
              {form.fields.length < MAX_FIELDS && (
                <button
                  type="button"
                  className="text-sm text-accent hover:underline"
                  onClick={() =>
                    setForm((f) => ({
                      ...f,
                      fields: [
                        ...f.fields,
                        blankField(
                          "fgABCEJ".split("").find((n) => !f.fields.some((x) => x.name === n)) ??
                            `F${f.fields.length + 1}`,
                          "scalar",
                        ),
                      ],
                    }))
                  }
                >
                  Add a field
                </button>
              )}
            </fieldset>

            <label className="block text-xs text-muted">
              Constants (names, separated by commas)
              <input
                aria-label="Constants"
                placeholder="k, mu0"
                className={`${FIELD} mt-0.5 block w-full`}
                value={form.constants}
                onChange={(e) => setForm((f) => ({ ...f, constants: e.target.value }))}
              />
            </label>
            <AssumptionsEditor
              value={form.assumptions}
              onChange={(assumptions) => setForm((f) => ({ ...f, assumptions }))}
            />
          </>
        )}

        {form.action === "apply_identity" && (
          <div className="flex flex-wrap gap-3">
            <label className="text-xs text-muted">
              Identity
              <input
                aria-label="Identity"
                placeholder="vector.curl_curl"
                className={`${FIELD} mt-0.5 block w-52`}
                value={form.ruleId}
                onChange={(e) => setForm((f) => ({ ...f, ruleId: e.target.value }))}
              />
            </label>
            <label className="text-xs text-muted">
              Path to the part
              <input
                aria-label="Path"
                placeholder="0, 1"
                className={`${FIELD} mt-0.5 block w-28`}
                value={form.path}
                onChange={(e) => setForm((f) => ({ ...f, path: e.target.value }))}
              />
            </label>
            <label className="flex items-center gap-1.5 self-end text-sm">
              <input
                type="checkbox"
                checked={form.reverse}
                onChange={(e) => setForm((f) => ({ ...f, reverse: e.target.checked }))}
              />
              right to left
            </label>
          </div>
        )}
        {form.action === "derive" && (
          <label className="block text-xs text-muted">
            Expression to reach
            <input
              aria-label="Target"
              className={`${FIELD} mt-0.5 block w-full`}
              value={form.target}
              onChange={(e) => setForm((f) => ({ ...f, target: e.target.value }))}
            />
          </label>
        )}

        {(form.action === "substitute" || form.action === "derive" || em) && (
          <fieldset className="space-y-2">
            <legend className="text-xs uppercase tracking-wide text-muted">
              Field equations you adopt as premises
            </legend>
            <div className="flex flex-wrap gap-x-4 gap-y-1">
              {PREMISE_OPTIONS.map((o) => (
                <label key={o.id} className="flex items-center gap-1.5 text-sm" title={o.equation}>
                  <input
                    type="checkbox"
                    checked={form.premiseIds.includes(o.id)}
                    onChange={() => toggleAdopted(o.id)}
                  />
                  {o.label}
                </label>
              ))}
            </div>
            <label className="block text-xs text-muted">
              Further equations of your own, one per line (lhs = rhs)
              <textarea
                aria-label="Premises"
                rows={2}
                className={`${FIELD} mt-0.5 block w-full`}
                value={form.premises}
                onChange={(e) => setForm((f) => ({ ...f, premises: e.target.value }))}
              />
            </label>
          </fieldset>
        )}

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
        <div key={run.key} className="space-y-2">
          <VectorPanel result={run.result} request={run.request} />
          {run.result.status === "ok" &&
            run.result.result_latex &&
            run.result.evidence !== "conditional" && (
              <button
                type="button"
                onClick={() => continueFrom(run.result)}
                className="text-sm text-accent hover:underline"
              >
                Use this result as the expression of the next step
              </button>
            )}
        </div>
      ))}
    </div>
  );
}
