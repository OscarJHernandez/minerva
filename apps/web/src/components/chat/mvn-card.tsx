import { CopyMath as Formula } from "@/components/copyable-math";
import type { MvnProvenance, MvnReason, MvnResult } from "@/lib/chat-types";
import { StatusPill } from "../result-panel";

/** A refusal in plain words, one sentence for each reason the engine can give. */
export const MVN_REASON_TEXT: Record<MvnReason, string> = {
  dimension_unsupported:
    "Only a Normal distribution of two components is supported: give a mean of two entries and a 2 by 2 covariance.",
  not_symmetric:
    "The covariance matrix is not symmetric, so it is not a covariance. Minerva does not symmetrise it for you.",
  not_positive_definite:
    "The covariance matrix is not positive definite, so it describes no Normal distribution.",
  singular_covariance:
    "The covariance matrix is singular, so the distribution has no density. Minerva does not use a pseudoinverse.",
  map_singular:
    "The linear map is singular: its image is a degenerate distribution without a density, which is not supported.",
  variables_missing: "The two components need names (for example x and y) for this operation.",
  index_invalid:
    "The component named is not one of the two components, or the value to condition on was not given for exactly one of them.",
  parse_error: "An entry could not be read, or the input is incomplete.",
  verification_failed:
    "The result did not pass its independent numerical check, so it is not shown.",
  resource_limit: "The computation reached a resource limit.",
};

const HEADING: Record<MvnResult["action"], string> = {
  density: "Density",
  moments: "Moments",
  marginal: "Marginal distribution",
  conditional: "Conditional distribution",
  affine: "Linear image",
};

function column(entries: string[]): string {
  return entries.length === 1
    ? (entries[0] ?? "")
    : `\\begin{pmatrix}${entries.join(" \\\\ ")}\\end{pmatrix}`;
}

function matrix(rows: string[][]): string {
  const first = rows[0] ?? [];
  if (rows.length === 1 && first.length === 1) return first[0] ?? "";
  return `\\begin{pmatrix}${rows.map((r) => r.join(" & ")).join(" \\\\ ")}\\end{pmatrix}`;
}

function checkText(c: MvnResult["checks"][number]): string {
  return `${c.method}: ${c.points_agreed} of ${c.points_checked} agree${c.passed ? "" : " (FAILED)"}`;
}

/**
 * The bivariate Normal: the mean and covariance a result describes, its density or correlation, and how it
 * was checked. Everything drawn comes from the engine's result, which exists only if quadrature of the printed
 * answer agreed; the evidence is worded as numerical ("checked", never "proved"). A result that holds given a
 * condition the declared assumptions cannot decide says so above the result. A mean and covariance the user
 * never wrote are labelled as the assistant's.
 */
export function MvnCard({
  result,
  provenance,
  given,
  headingId = "mvn-heading",
}: {
  result: MvnResult;
  provenance?: MvnProvenance;
  /** conditional only: the component fixed and its value, as the request stated them */
  given?: Record<string, string>;
  headingId?: string;
}) {
  const shown = result.status === "ok";
  const conditional = shown && (result.evidence === "conditional" || result.premises.length > 0);
  const open = result.conditions.filter((c) => !c.established);
  const names = result.variables.join(", ");
  const oneDimensional = result.mean_latex.length === 1;
  const reason = result.reason ? MVN_REASON_TEXT[result.reason] : null;
  const label = provenance && provenance.who !== "user" ? provenance.label : null;

  return (
    <section aria-labelledby={headingId} className="rounded-lg border border-rule bg-raised p-5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 id={headingId} className="font-serif text-xl">
          Bivariate Normal: {HEADING[result.action].toLowerCase()}
        </h3>
        <StatusPill
          status={result.status}
          okLabel={conditional ? "Holds given a premise" : "Numerically checked"}
        />
      </div>

      {label && (
        <div
          role="note"
          data-testid="assumed-note"
          className="mt-3 rounded-md border border-warn/50 bg-warn-soft/40 px-3 py-2 text-xs"
        >
          <p className="font-medium text-warn">{label.charAt(0).toUpperCase() + label.slice(1)}</p>
          {provenance && provenance.values.length > 0 && (
            <p className="mt-1 text-muted">Not written by you: {provenance.values.join(", ")}</p>
          )}
          <p className="mt-1 text-muted">
            The result holds for these values. Confirm or change them.
          </p>
        </div>
      )}

      {!shown && (
        <div className="mt-3 rounded bg-warn-soft p-3 text-sm text-warn" data-testid="refusal">
          {reason && <p>{reason}</p>}
          {result.message && <p className={reason ? "mt-1 text-xs" : ""}>{result.message}</p>}
          <p className="mt-1 text-xs">No result is shown.</p>
        </div>
      )}

      {conditional && (
        <div
          className="mt-3 rounded border border-rule bg-sunk px-3 py-2 text-sm"
          data-testid="premises"
        >
          <div className="text-xs uppercase tracking-wide text-muted">
            Holds only given (Minerva does not check these)
          </div>
          <ul className="mt-1 space-y-1">
            {open.length > 0
              ? open.map((c, i) => (
                  <li key={`${i}-${c.latex}`} className="min-w-0">
                    <span className="overflow-x-auto">
                      <Formula latex={c.latex} />
                    </span>
                    <span className="text-muted"> · {c.how}</span>
                  </li>
                ))
              : result.premises.map((p) => <li key={p}>{p}</li>)}
          </ul>
        </div>
      )}

      {shown && (
        <div className="mt-3 space-y-1" data-testid="distribution">
          {/* not uppercase: the component names are case-sensitive (x is not X) */}
          <div className="text-xs text-muted">
            {result.action === "density" || result.action === "moments"
              ? `Distribution of (${names})`
              : result.action === "conditional" && given
                ? `Distribution of ${names} given ${Object.entries(given)
                    .map(([k, v]) => `${k} = ${v}`)
                    .join(", ")}`
                : `Distribution of ${names}`}
          </div>
          <div className="overflow-x-auto">
            <Formula latex={`\\mu = ${column(result.mean_latex)}`} display label="Mean" />
          </div>
          <div className="overflow-x-auto">
            <Formula
              latex={`${oneDimensional ? "\\sigma^{2}" : "\\Sigma"} = ${matrix(result.covariance_latex)}`}
              display
              label="Covariance"
            />
          </div>
          {result.density_latex && (
            <div className="overflow-x-auto">
              <Formula latex={`f(${names}) = ${result.density_latex}`} display label="Density" />
            </div>
          )}
          {result.correlation_latex && (
            <div className="overflow-x-auto">
              <Formula latex={`\\rho = ${result.correlation_latex}`} display label="Correlation" />
            </div>
          )}
        </div>
      )}

      {shown && (
        <div className="mt-3 text-xs text-muted" data-testid="checks">
          <p>Checked numerically by quadrature of the printed result: evidence, not a proof.</p>
          <ul className="mt-1 list-disc pl-5">
            {result.checks.map((c, i) => (
              <li key={`${i}-${c.kind}`} className={c.passed ? "" : "text-bad"}>
                {checkText(c)}
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
