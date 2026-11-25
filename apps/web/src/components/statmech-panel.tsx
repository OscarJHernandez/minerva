"use client";

import { CopyMath as Math } from "@/components/copyable-math";
import { useId, useState } from "react";
import type {
  ResultKind,
  StatAnswers,
  StatCheck,
  StatMechResult,
  StatQuantity,
  StatStep,
} from "@/lib/statmech-types";
import symbols from "../../../../packages/ui/fixtures/statmech-symbols.json";
import { occupationKey, splitOccupations } from "@/lib/statmech-occupations";
import { Numbered } from "./chat/equation-number";
import { LinePlot, type PlotSeries } from "./chat/line-plot";
import { StatusPill } from "./result-panel";

const KIND_LABEL: Record<ResultKind, string> = {
  exact: "exact",
  representation: "unevaluated representation",
  numeric: "numerical",
  asymptotic: "asymptotic",
};

const REASON_TEXT: Record<string, string> = {
  divergent: "the series diverges",
  convergence_not_provable: "convergence could not be proved",
  not_separable: "the Hamiltonian does not separate",
  convention_unconfirmed: "a convention is needed from you",
  invalid_degeneracy: "a degeneracy is not a positive integer",
  assumptions_insufficient: "the declared assumptions are not enough",
  ensemble_mismatch: "this ensemble cannot hold those variables fixed",
  coupling_not_linear: "the coupling is not linear",
  unknown_observable: "unknown observable",
  verification_failed: "the independent check disagreed",
  unsupported_system: "not supported",
  resource_limit: "resource limit",
  parse_error: "could not read the input",
};

/**
 * How a base quantity is written: the one closed table shared with the server
 * (`packages/ui/fixtures/statmech-symbols.json`, `card_equations.STATMECH_SYMBOL`). A name not in it is shown as
 * the engine named it.
 */
const SYMBOL: Record<string, string> = symbols.symbols;
const ORDER = "<order>";

/** A quantity the user labelled (a partial derivative) is never drawn with a table symbol, whatever its label. */
export function quantitySymbol(q: {
  name: string;
  variable: string | null;
  order?: number | null;
}): string {
  if (q.variable !== null) return q.name;
  const symbol = SYMBOL[q.name];
  if (symbol === undefined) return q.name;
  if (!symbol.includes(ORDER)) return symbol;
  // a cumulant symbol needs the engine's own order (an integer 1..4); without it the name is shown, as the server does
  const order = q.order;
  return typeof order === "number" && Number.isInteger(order) && order >= 1 && order <= 4
    ? symbol.replace(ORDER, String(order))
    : q.name;
}

/**
 * The key the server numbers a quantity under (`card_equations.statmech_key`): `quantity:<name>`, and
 * `quantity:<name>:<k>` for the k-th occurrence of a name (k >= 2).
 */
export function quantityKey(
  quantities: { name: string; order?: number | null }[],
  i: number,
): string {
  const name = quantities[i]?.name ?? "";
  const order = quantities[i]?.order ?? null;
  if (order !== null && name.startsWith("cumulant_")) {
    // a cumulant's key carries its order (`cumulant_E:3`, `cumulant_E:3:2` for the second request of it)
    const n = quantities.slice(0, i + 1).filter((q) => q.name === name && q.order === order).length;
    return n === 1 ? `${name}:${order}` : `${name}:${order}:${n}`;
  }
  const k = quantities.slice(0, i + 1).filter((q) => q.name === name).length;
  return k === 1 ? `quantity:${name}` : `quantity:${name}:${k}`;
}

/**
 * The rows of the occupation list, each under the key the server numbers it by (`card_equations.occupation_key`);
 * null for any other quantity, for a user-labelled one and for a text that does not split exactly.
 */
function occupationRows(
  quantities: StatQuantity[],
  i: number,
): { key: string; latex: string }[] | null {
  const q = quantities[i];
  if (q === undefined || q.name !== "occupations" || q.variable !== null) return null;
  const pieces = splitOccupations(q.latex);
  if (pieces === null) return null;
  return pieces.map((latex, j) => ({ key: occupationKey(quantities, i, j + 1), latex }));
}

function checkLabel(c: StatCheck): string {
  const state = c.passed === true ? "agreed" : c.passed === false ? "DISAGREED" : "undecided";
  const count = c.points > 0 ? ` (${c.agreed}/${c.points} points)` : "";
  return `${c.name.replace(/_/g, " ")}: ${state}${count}`;
}

function Step({ step }: { step: StatStep }) {
  return (
    <li className="min-w-0" data-testid={`step-${step.name}`}>
      <div className="flex flex-wrap items-baseline gap-2">
        <h4 className="text-xs uppercase tracking-wide text-muted">{step.title}</h4>
        {step.result_kind !== "exact" && (
          <span className="rounded bg-sunk px-1.5 py-0.5 text-xs text-warn">
            {KIND_LABEL[step.result_kind]}
          </span>
        )}
      </div>
      <div className="overflow-x-auto">
        <Math latex={step.latex} display label={step.title} />
      </div>
      {step.conditions.length > 0 && (
        <ul className="mt-1 space-y-0.5 text-xs text-muted">
          {step.conditions.map((c, i) => (
            <li key={`${c.latex}-${i}`}>
              <span className={c.established ? "text-ok" : "text-warn"}>
                {c.established ? "holds" : "not established"}
              </span>{" "}
              <Math latex={c.latex} /> — {c.how}
            </li>
          ))}
        </ul>
      )}
    </li>
  );
}

function Quantity({
  q,
  k,
  occupationKeys,
}: {
  q: StatQuantity;
  k: string;
  /** Row keys of the per-state occupations, when this quantity is the occupation list and it splits exactly. */
  occupationKeys: { key: string; latex: string }[] | null;
}) {
  return (
    <section
      aria-label={q.title}
      className="min-w-0 rounded border border-rule bg-sunk p-3"
      data-testid={`quantity-${q.name}`}
    >
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="text-sm font-medium">{q.title}</h3>
        <span className="text-xs text-muted">{KIND_LABEL[q.result_kind]}</span>
      </div>
      {occupationKeys !== null ? (
        occupationKeys.map((row) => (
          <Numbered key={row.key} k={row.key}>
            <div className="overflow-x-auto">
              <Math latex={row.latex} display label={q.title} />
            </div>
          </Numbered>
        ))
      ) : (
        <Numbered k={k}>
          <div className="overflow-x-auto">
            <Math latex={`${quantitySymbol(q)} = ${q.latex}`} display label={q.title} />
          </div>
        </Numbered>
      )}
      {q.held_fixed.length > 0 && (
        <p className="text-xs text-muted">Held fixed: {q.held_fixed.join(", ")}</p>
      )}
      {q.route && <p className="text-xs text-muted">{q.route}</p>}
      {q.checks.length > 0 && (
        <ul className="mt-1 text-xs text-muted" aria-label="Independent checks">
          {q.checks.map((c, i) => (
            <li key={`${c.name}-${i}`} title={c.detail}>
              {checkLabel(c)}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

/** Series for the sweep plot: the first column is the swept variable, the others are quantities. */
export function sweepSeries(table: NonNullable<StatMechResult["table"]>): PlotSeries[] {
  const x = table.rows.map((r) => Number(r[0]));
  return table.columns.slice(1).map((label, j) => ({
    label,
    x,
    y: table.rows.map((r) => Number(r[j + 1])),
  }));
}

/** Which of the user's choices the engine still needs, read from the question it asked. */
function Questions({
  result,
  needs,
  onAnswer,
}: {
  result: StatMechResult;
  needs: { cell: boolean; identical: boolean };
  onAnswer?: (answers: StatAnswers) => Promise<void> | void;
}) {
  const id = useId();
  const [cell, setCell] = useState("");
  const [identical, setIdentical] = useState<"yes" | "no" | "">("");
  const ready = (!needs.cell || cell.trim() !== "") && (!needs.identical || identical !== "");
  return (
    <form
      aria-labelledby={`${id}-q`}
      className="mt-3 space-y-3 rounded border border-warn p-3"
      onSubmit={(e) => {
        e.preventDefault();
        if (!ready || !onAnswer) return;
        void onAnswer({
          ...(needs.cell ? { phase_space_cell: cell.trim() } : {}),
          ...(needs.identical ? { identical: identical === "yes" } : {}),
        });
      }}
    >
      <h3 id={`${id}-q`} className="text-sm font-medium">
        This is your choice
      </h3>
      {result.questions.map((q) => (
        <p key={q} className="text-sm">
          {q}
        </p>
      ))}
      {needs.cell && (
        <label className="block text-sm">
          Phase-space cell
          <input
            className="mt-1 block w-full rounded border border-rule bg-raised px-2 py-1.5 font-mono text-sm"
            value={cell}
            placeholder="h^3"
            onChange={(e) => setCell(e.target.value)}
          />
        </label>
      )}
      {needs.identical && (
        <fieldset className="text-sm">
          <legend>The copies are</legend>
          <label className="mr-4">
            <input
              type="radio"
              name={`${id}-identical`}
              checked={identical === "yes"}
              onChange={() => setIdentical("yes")}
            />{" "}
            indistinguishable (divide by N!)
          </label>
          <label>
            <input
              type="radio"
              name={`${id}-identical`}
              checked={identical === "no"}
              onChange={() => setIdentical("no")}
            />{" "}
            distinguishable
          </label>
        </fieldset>
      )}
      {onAnswer ? (
        <button
          type="submit"
          disabled={!ready}
          className="rounded bg-accent px-3 py-1.5 text-sm font-medium text-accent-ink disabled:opacity-50"
        >
          Continue with this choice
        </button>
      ) : (
        <p className="text-xs text-muted">Open a conversation or the workbench to answer.</p>
      )}
    </form>
  );
}

export function StatMechPanel({
  result,
  needs = { cell: false, identical: false },
  onAnswer,
}: {
  result: StatMechResult;
  /** Which choices the question is about (known from the request that was made). */
  needs?: { cell: boolean; identical: boolean };
  onAnswer?: (answers: StatAnswers) => Promise<void> | void;
}) {
  const ok = result.status === "ok";
  const v = result.verification;
  const okLabel =
    result.action === "analyze"
      ? "Methods"
      : v.symbolic_zero
        ? "Checked numerically; symbolic zero found for some quantities"
        : result.action === "evaluate"
          ? "Numerical values, checked"
          : "Numerically checked";
  const plot = result.table ? sweepSeries(result.table) : [];
  return (
    <section
      aria-label="Statistical mechanics"
      className="min-w-0 rounded-lg border border-rule bg-raised p-4"
      data-testid="statmech-panel"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="font-serif text-lg">Statistical mechanics</h3>
        <StatusPill status={result.status} okLabel={okLabel} />
      </div>

      {!ok && (
        <div role="status" className="mt-2 text-sm">
          <p>{result.message}</p>
          {result.reason && (
            <p className="text-xs text-muted">
              Reason: {REASON_TEXT[result.reason] ?? result.reason}. Nothing is shown as a result.
            </p>
          )}
        </div>
      )}

      {result.reason === "convention_unconfirmed" && (
        <Questions result={result} needs={needs} onAnswer={onAnswer} />
      )}

      {ok && result.steps.length > 0 && (
        <ol className="mt-3 space-y-3" aria-label="Derivation">
          {result.steps.map((s, i) => (
            <Step key={`${s.name}-${i}`} step={s} />
          ))}
        </ol>
      )}

      {ok && result.quantities.length > 0 && (
        <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-2">
          {result.quantities.map((q, i) => (
            <Quantity
              key={`${q.name}-${i}`}
              q={q}
              k={quantityKey(result.quantities, i)}
              occupationKeys={occupationRows(result.quantities, i)}
            />
          ))}
        </div>
      )}

      {ok && result.table && (
        <div className="mt-4 space-y-3">
          {plot.length > 0 && result.table.rows.length >= 2 && (
            <LinePlot
              series={plot}
              xLabel={result.table.variable}
              yLabel="value"
              label={`Sweep over ${result.table.variable}`}
            />
          )}
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr>
                  {result.table.columns.map((c) => (
                    <th key={c} className="whitespace-nowrap pr-3 font-medium">
                      {c}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {result.table.rows.map((row, i) => (
                  <tr key={i}>
                    {row.map((cell, j) => (
                      <td key={j} className="whitespace-nowrap pr-3 font-mono">
                        {cell}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {ok && result.approximation && (
        <section
          aria-label="Approximation"
          className="mt-4 rounded border border-warn p-3"
          data-testid="statmech-approximation"
        >
          <h4 className="text-sm font-medium">
            An approximation, not an identity ({result.approximation.method.replace(/_/g, " ")})
          </h4>
          <dl className="mt-2 space-y-1 text-sm">
            <div>
              <dt className="inline text-muted">Expansion parameter: </dt>
              <dd className="inline">{result.approximation.parameter}</dd>
            </div>
            <div>
              <dt className="inline text-muted">Expected accuracy: </dt>
              <dd className="inline">{result.approximation.order}</dd>
            </div>
            <div>
              <dt className="inline text-muted">Valid: </dt>
              <dd className="inline">{result.approximation.validity}</dd>
            </div>
          </dl>
          <p className="mt-2 text-xs text-muted">Assumptions</p>
          <ul className="list-disc pl-5 text-sm">
            {result.approximation.assumptions.map((a) => (
              <li key={a}>{a}</li>
            ))}
          </ul>
          {result.approximation.error_table.length > 0 && (
            <div className="mt-2 overflow-x-auto">
              <table className="w-full text-left text-xs">
                <caption className="mb-1 text-left text-muted">
                  Measured against the independent evaluation (a measurement, not a bound)
                </caption>
                <thead>
                  <tr>
                    {["", "exact", "approximate", "error", ""].map((h, i) => (
                      <th key={i} className="whitespace-nowrap pr-3 font-medium">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.approximation.error_table.map((row) => (
                    <tr key={row.label}>
                      <td className="whitespace-nowrap pr-3">{row.label}</td>
                      <td className="whitespace-nowrap pr-3 font-mono">{row.exact}</td>
                      <td className="whitespace-nowrap pr-3 font-mono">{row.approximate}</td>
                      <td className="whitespace-nowrap pr-3 font-mono">{row.error}</td>
                      <td className="whitespace-nowrap text-muted">{row.note}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      )}

      {result.methods.length > 0 && (
        <ul className="mt-3 space-y-2 text-sm" aria-label="Methods">
          {result.methods.map((m) => (
            <li key={m.id} data-testid={`method-${m.id}`}>
              <span className="font-medium">{m.title}</span>{" "}
              <span className={m.applies ? "text-ok" : "text-muted"}>
                {m.applies ? "applies" : "does not apply"}
              </span>
              {m.applies && (
                <span className={m.implemented ? "text-ok" : "text-warn"}>
                  {m.implemented ? ", implemented" : ", not implemented yet"}
                </span>
              )}
              <p className="text-xs text-muted">{m.why}</p>
            </li>
          ))}
        </ul>
      )}
      {result.not_supported.length > 0 && (
        <div className="mt-3 text-xs text-muted">
          <p className="font-medium">Not supported</p>
          <ul className="list-disc pl-5">
            {result.not_supported.map((n) => (
              <li key={n}>{n}</li>
            ))}
          </ul>
        </div>
      )}

      {ok && result.notes.length > 0 && (
        <ul className="mt-3 list-disc pl-5 text-xs text-muted">
          {result.notes.map((n) => (
            <li key={n}>{n}</li>
          ))}
        </ul>
      )}

      {ok && v.passed && (
        <p className="mt-3 text-xs text-muted" data-testid="statmech-evidence">
          {v.points_agreed}/{v.points_checked} points agreed with an independent evaluation.{" "}
          {v.note}.
        </p>
      )}
    </section>
  );
}
