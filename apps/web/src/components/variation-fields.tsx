"use client";

import {
  MAX_CONDITIONS,
  blankCondition,
  functionNames,
  parseGiven,
  type ConditionForm,
  type VariationForm,
} from "@/lib/variation-form";

export const FIELD =
  "min-w-0 rounded border border-rule bg-raised px-2 py-1.5 font-mono text-sm text-ink disabled:opacity-50";

type SetForm = (update: (f: VariationForm) => VariationForm) => void;

/**
 * The endpoint conditions as a table. Nothing is pre-filled: a derivative that is not listed is left
 * undecided, and its boundary term is kept by the server.
 */
export function ConditionsEditor({ form, setForm }: { form: VariationForm; setForm: SetForm }) {
  function update(i: number, next: Partial<ConditionForm>) {
    setForm((f) => ({
      ...f,
      conditions: f.conditions.map((c, k) => (k === i ? { ...c, ...next } : c)),
    }));
  }

  return (
    <fieldset className="min-w-0 space-y-2">
      <legend className="text-xs uppercase tracking-wide text-muted">Endpoint conditions</legend>
      <p className="text-xs text-muted">
        State what holds for each derivative at each end. One you do not list stays undecided and
        its boundary term is kept.
      </p>
      {form.conditions.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full min-w-max text-left text-sm" aria-label="Endpoint conditions">
            <thead>
              <tr className="text-xs uppercase tracking-wide text-muted">
                {["Function", "Derivative", "End", "Kind", "Value", ""].map((h) => (
                  <th key={h} scope="col" className="px-1 py-1 font-normal">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {form.conditions.map((c, i) => (
                <tr key={i}>
                  <td className="px-1 py-1">
                    <input
                      aria-label={`Condition ${i + 1} function`}
                      className={`${FIELD} w-20`}
                      value={c.function}
                      onChange={(e) => update(i, { function: e.target.value })}
                    />
                  </td>
                  <td className="px-1 py-1">
                    <input
                      aria-label={`Condition ${i + 1} derivative order`}
                      inputMode="numeric"
                      className={`${FIELD} w-16`}
                      value={c.order}
                      onChange={(e) => update(i, { order: e.target.value })}
                    />
                  </td>
                  <td className="px-1 py-1">
                    <select
                      aria-label={`Condition ${i + 1} end`}
                      className={FIELD}
                      value={c.end}
                      onChange={(e) => update(i, { end: e.target.value as "lower" | "upper" })}
                    >
                      <option value="lower">lower</option>
                      <option value="upper">upper</option>
                    </select>
                  </td>
                  <td className="px-1 py-1">
                    <select
                      aria-label={`Condition ${i + 1} kind`}
                      className={FIELD}
                      value={c.kind}
                      onChange={(e) => update(i, { kind: e.target.value as "fixed" | "free" })}
                    >
                      <option value="fixed">fixed</option>
                      <option value="free">free</option>
                    </select>
                  </td>
                  <td className="px-1 py-1">
                    <input
                      aria-label={`Condition ${i + 1} value`}
                      className={`${FIELD} w-24`}
                      disabled={c.kind === "free"}
                      placeholder={c.kind === "free" ? "" : "value"}
                      value={c.kind === "free" ? "" : c.value}
                      onChange={(e) => update(i, { value: e.target.value })}
                    />
                  </td>
                  <td className="px-1 py-1">
                    <button
                      type="button"
                      aria-label={`Remove condition ${i + 1}`}
                      className="text-xs text-muted hover:text-bad"
                      onClick={() =>
                        setForm((f) => ({
                          ...f,
                          conditions: f.conditions.filter((_, k) => k !== i),
                        }))
                      }
                    >
                      ✕
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <button
        type="button"
        disabled={form.conditions.length >= MAX_CONDITIONS}
        className="text-sm text-accent hover:underline disabled:opacity-50"
        onClick={() =>
          setForm((f) => ({
            ...f,
            conditions: [
              ...f.conditions,
              { ...blankCondition(), function: functionNames(f)[0] ?? "" },
            ],
          }))
        }
      >
        Add a condition
      </button>
    </fieldset>
  );
}

function setIn(
  form: VariationForm,
  key: "values" | "definitions" | "paths",
  name: string,
  value: string,
): VariationForm {
  return { ...form, [key]: { ...form[key], [name]: value } };
}

/**
 * The numbers only the user can choose: parameter values, definitions of given functions, one trial
 * path (or starting guess) per function, an optional one-parameter family. All start empty.
 */
export function NumbersFields({
  form,
  setForm,
  parameters,
}: {
  form: VariationForm;
  setForm: SetForm;
  /** The parameters the server found in the functional (empty before a first result). */
  parameters: string[];
}) {
  const functions = functionNames(form);
  const given = parseGiven(form.givenFunctions);
  const givenList = typeof given === "string" ? [] : given;
  return (
    <div className="space-y-3">
      {parameters.length > 0 ? (
        <div>
          <div className="text-xs uppercase tracking-wide text-muted">Parameter values</div>
          <div className="mt-1 flex flex-wrap gap-3">
            {parameters.map((p) => (
              <label key={p} className="text-xs text-muted">
                {p}
                <input
                  aria-label={`Value of ${p}`}
                  className={`${FIELD} mt-0.5 block w-24`}
                  value={form.values[p] ?? ""}
                  onChange={(e) => setForm((f) => setIn(f, "values", p, e.target.value))}
                />
              </label>
            ))}
          </div>
        </div>
      ) : (
        <p className="text-xs text-muted">
          Run “Analyze” or “First variation” to list the parameters that need values.
        </p>
      )}
      {givenList.length > 0 && (
        <div className="space-y-2">
          <div className="text-xs uppercase tracking-wide text-muted">Given functions</div>
          {givenList.map((g) => (
            <label key={g.name} className="block text-xs text-muted">
              {g.name}({g.args.join(", ")}) =
              <input
                aria-label={`Definition of ${g.name}`}
                className={`${FIELD} mt-0.5 block w-full`}
                placeholder={`expression in ${g.args.join(", ")}`}
                value={form.definitions[g.name] ?? ""}
                onChange={(e) => setForm((f) => setIn(f, "definitions", g.name, e.target.value))}
              />
            </label>
          ))}
        </div>
      )}
      <div className="space-y-2">
        <div className="text-xs uppercase tracking-wide text-muted">
          Trial path or starting guess
        </div>
        {functions.map((fn) => (
          <label key={fn} className="block text-xs text-muted">
            {fn}({form.variable || "t"}) =
            <input
              aria-label={`Path for ${fn}`}
              className={`${FIELD} mt-0.5 block w-full`}
              placeholder={`expression in ${form.variable || "t"}, e.g. t^2`}
              value={form.paths[fn] ?? ""}
              onChange={(e) => setForm((f) => setIn(f, "paths", fn, e.target.value))}
            />
          </label>
        ))}
        <div className="flex flex-wrap gap-3">
          <label className="text-xs text-muted">
            Family parameter (optional)
            <input
              aria-label="Family parameter"
              className={`${FIELD} mt-0.5 block w-24`}
              value={form.sweepName}
              onChange={(e) => setForm((f) => ({ ...f, sweepName: e.target.value }))}
            />
          </label>
          <label className="min-w-0 flex-1 text-xs text-muted">
            Its values, 2 to 6, separated by commas
            <input
              aria-label="Family values"
              className={`${FIELD} mt-0.5 block w-full`}
              value={form.sweepValues}
              onChange={(e) => setForm((f) => ({ ...f, sweepValues: e.target.value }))}
            />
          </label>
        </div>
      </div>
    </div>
  );
}
