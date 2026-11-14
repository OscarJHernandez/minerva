"use client";

import { Math } from "@minerva/ui/math";
import { instanceWording } from "@/lib/derivation";
import type {
  ExpectationResult,
  ResultCondition,
  StatEstimate,
  StatModelResult,
  StatStep,
  Status,
  SumsResult,
} from "@/lib/types";
import { StatusPill } from "./result-panel";

export type StatisticsKind = "sums" | "expectation" | "statistics";

/** A refusal in plain words. The enumerated reason comes from the engine; the message adds the detail. */
export const REASON_TEXT: Record<string, string> = {
  unsupported_family: "Minerva does not know this family of distributions.",
  parameter_invalid: "A parameter lies outside the parameter space of the model.",
  parameter_missing: "A parameter has no value, and it is not one of the parameters to estimate.",
  assumptions_insufficient: "The declared assumptions are not enough to decide this.",
  independence_not_declared:
    "Several observations were given without saying they are independent. Minerva never assumes independence.",
  regularity_not_satisfied:
    "The support depends on the parameter, so there is no score or Fisher information here.",
  convergence_not_established:
    "The integral does not converge absolutely, so the moment is not shown (a principal value is not a mean).",
  no_closed_form: "No closed form was found, and none is made up.",
  system_not_triangular:
    "The score equations cannot be solved one parameter at a time; only triangular systems are solved.",
  candidate_inadmissible:
    "For these data the maximum would lie outside the parameter space, so there is no maximum-likelihood estimate.",
  kernel_not_recognised:
    "Prior times likelihood is not the kernel of a family Minerva knows, so no posterior is shown.",
  verification_failed: "The result did not pass its independent check, so it is not shown.",
  parse_error: "The input could not be read.",
  unsupported_form: "This form is not supported.",
  resource_limit: "The computation reached a resource limit.",
  range_not_provable:
    "Minerva cannot show that the range of the sum is non-empty: declare its upper bound a positive integer.",
  conditions_unmet: "A condition of this rewrite is not established by the declared assumptions.",
  missing_candidate: "There is no answer to check.",
  no_random_variables: "No random variable was declared.",
};

const GREEK = new Set([
  "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
  "lambda", "mu", "nu", "xi", "rho", "sigma", "tau", "phi", "chi", "psi", "omega",
]); // prettier-ignore

/** A parameter name as LaTeX: mu -> \mu, lam -> \lambda, v -> v. */
export function nameLatex(name: string): string {
  if (name === "lam") return "\\lambda";
  return GREEK.has(name) ? `\\${name}` : name;
}

/** The engine words some conditions as prose ("mu is a real number"): keep the prose out of math italics. */
export function conditionLatex(latex: string): string {
  const m = /^(\S+) is (an? .+)$/.exec(latex);
  return m ? `${nameLatex(m[1] ?? "")}\\text{ is ${m[2]}}` : latex;
}

function matrixLatex(rows: string[][]): string {
  return `\\begin{pmatrix}${rows.map((r) => r.join(" & ")).join(" \\\\ ")}\\end{pmatrix}`;
}

const FAMILY_TITLE: Record<string, string> = {
  normal: "Normal",
  beta: "Beta",
  gamma: "Gamma (shape and rate)",
  bernoulli: "Bernoulli",
  binomial: "Binomial",
  poisson: "Poisson",
  cauchy: "Cauchy",
  uniform_0_theta: "Uniform on (0, θ)",
};

const MOMENT_LABEL: Record<string, string> = {
  mean: "\\mathbb{E}[X]",
  variance: "\\operatorname{Var}(X)",
  second: "\\mathbb{E}[X^{2}]",
};

function expressionLabel(key: string): string {
  if (key === "loglik") return "Log-likelihood";
  if (key.startsWith("score:")) return `Score for ${key.slice(6)}`;
  return key;
}

/** What an estimate's optimum rests on. "Proved" only when the engine says it proved it. */
export function optimumText(e: StatEstimate): string | null {
  if (e.optimum === "global") {
    return e.optimum_evidence === "proved"
      ? "Global maximum: the log-likelihood is concave on the whole parameter space (proved)."
      : "Global maximum.";
  }
  if (e.optimum === "local") {
    return e.optimum_evidence === "proved"
      ? "Local maximum: the Hessian is negative definite at the estimate (proved, given the condition below). The log-likelihood is not shown to be concave, so this is not claimed to be global."
      : "Local maximum: the Hessian is negative definite on explicit data (measured, not proved). Not claimed to be global.";
  }
  if (e.optimum === "stationary_only") return "A stationary point; its type was not established.";
  return null;
}

function Conditions({ items, title }: { items: ResultCondition[]; title: string }) {
  if (items.length === 0) return null;
  return (
    <div className="mt-3" data-testid="conditions">
      <div className="text-xs uppercase tracking-wide text-muted">{title}</div>
      <ul className="mt-1 space-y-1 text-sm">
        {items.map((c, i) => (
          <li key={`${i}-${c.latex}`} className="min-w-0">
            <span className="overflow-x-auto">
              <Math latex={conditionLatex(c.latex)} />
            </span>{" "}
            <span className={c.established ? "text-ok" : "text-warn"}>
              {c.established ? "established" : "assumed"}
            </span>
            <span className="text-muted"> · {c.how}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function Steps({ steps }: { steps: StatStep[] }) {
  if (steps.length === 0) return null;
  return (
    <div className="mt-4">
      <h3 className="text-sm font-medium">
        Steps <span className="font-normal text-muted">({steps.length})</span>
      </h3>
      <ol className="mt-2 space-y-2">
        {steps.map((step, i) => (
          <li key={`${i}-${step.title}`} className="min-w-0 rounded border border-rule bg-sunk p-3">
            <div className="text-sm">
              {i + 1}. {step.title}
              {step.evidence === "conditional" && (
                <span className="ml-2 text-xs text-muted">(holds given the premises above)</span>
              )}
            </div>
            {step.before_latex && (
              <div className="overflow-x-auto">
                <Math latex={step.before_latex} display />
              </div>
            )}
            {step.after_latex && (
              <div className="overflow-x-auto">
                <Math
                  latex={
                    step.before_latex ? `\\Rightarrow\\; ${step.after_latex}` : step.after_latex
                  }
                  display
                />
              </div>
            )}
          </li>
        ))}
      </ol>
    </div>
  );
}

interface CheckLine {
  text: string;
  ok: boolean;
}

function checkLines(
  kind: StatisticsKind,
  result: SumsResult | ExpectationResult | StatModelResult,
) {
  const lines: CheckLine[] = [];
  if (kind === "sums") {
    const r = result as SumsResult;
    if (r.verification) {
      lines.push({ text: instanceWording(r.verification), ok: r.verification.passed });
    }
    if (r.derivative_verification) {
      const d = r.derivative_verification;
      lines.push({
        text: `independent numerical derivative on every instance: ${d.points_agreed}/${d.points_checked} points agree`,
        ok: d.passed,
      });
    }
  } else if (kind === "expectation") {
    const v = (result as ExpectationResult).verification;
    if (v) {
      lines.push({
        text: `agrees on ${v.models_agreed} of ${v.models_checked} explicit joint distributions (dependent unless independence was declared); evidence, not a proof for every distribution`,
        ok: v.passed,
      });
    }
  } else {
    for (const c of (result as StatModelResult).checks) {
      lines.push({
        text: `${c.method}: ${c.points_agreed} of ${c.points_checked} points agree`,
        ok: c.passed,
      });
    }
  }
  return lines;
}

function okLabel(
  kind: StatisticsKind,
  result: { evidence?: string | null; status: Status },
): string {
  const conditional = result.evidence === "conditional";
  if (kind === "sums") return "Checked on explicit instances";
  if (kind === "expectation") {
    return conditional ? "Holds given your independence" : "Checked on explicit distributions";
  }
  return conditional ? "Holds given your premises" : "Derived and checked";
}

function heading(kind: StatisticsKind, result: { action: string }): string {
  if (kind === "sums") return "Finite sums";
  if (kind === "expectation") return "Expectation, variance, covariance";
  const names: Record<string, string> = {
    moments: "Moments",
    log_likelihood: "Log-likelihood",
    score: "Score",
    mle: "Maximum likelihood",
    fisher_information: "Fisher information",
    posterior: "Posterior",
  };
  return names[result.action] ?? "Statistical model";
}

/**
 * Sums, expectations and statistical models. A formula is drawn only when the engine verified it and
 * the evidence is worded as what it is: an instance check says "checked for n = ...", a numerical
 * check says "points agree", and "proved" appears only where the engine reports a proof. A refusal
 * shows its reason in plain words and no formula. A premise the user stated (independence, a prior)
 * is shown above the result, because the result holds given it.
 */
export function StatisticsPanel({
  kind,
  result,
  headingId = "statistics-heading",
  continuations,
  onContinue,
}: {
  kind: StatisticsKind;
  result: SumsResult | ExpectationResult | StatModelResult;
  headingId?: string;
  /** For a statistical step: the expressions it can be continued from. */
  continuations?: { label: string }[];
  onContinue?: (index: number) => void;
}) {
  const shown = result.status === "ok" || result.status === "unchanged";
  const evidence = "evidence" in result ? result.evidence : null;
  const premises: string[] =
    kind === "expectation"
      ? (result as ExpectationResult).given
      : kind === "statistics"
        ? (result as StatModelResult).premises
        : [];
  const conditions: ResultCondition[] =
    "conditions" in result ? (result.conditions as ResultCondition[]) : [];
  const stat = kind === "statistics" ? (result as StatModelResult) : null;
  const single = kind !== "statistics" ? (result as SumsResult | ExpectationResult) : null;
  const lines = checkLines(kind, result);
  const reasonText = result.reason ? (REASON_TEXT[result.reason] ?? result.reason) : null;

  return (
    <section aria-labelledby={headingId} className="rounded-lg border border-rule bg-raised p-5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 id={headingId} className="font-serif text-xl">
          {heading(kind, result)}
        </h2>
        <StatusPill
          status={result.status}
          okLabel={okLabel(kind, { evidence, status: result.status })}
        />
      </div>

      {single?.input_latex && (
        <div className="mt-4">
          <div className="text-xs uppercase tracking-wide text-muted">Input as read</div>
          <div className="overflow-x-auto">
            <Math latex={single.input_latex} display />
          </div>
        </div>
      )}

      {shown && premises.length > 0 && (
        <div
          className="mt-3 rounded border border-rule bg-sunk px-3 py-2 text-sm"
          data-testid="premises"
        >
          <div className="text-xs uppercase tracking-wide text-muted">
            Holds given (stated by you, not checked by Minerva)
          </div>
          <ul className="mt-1 list-disc pl-5">
            {premises.map((p) => (
              <li key={p}>{p}</li>
            ))}
          </ul>
        </div>
      )}

      {!shown && (
        <div className="mt-3 rounded bg-warn-soft p-3 text-sm text-warn" data-testid="refusal">
          {reasonText && <p>{reasonText}</p>}
          {result.message && <p className={reasonText ? "mt-1 text-xs" : ""}>{result.message}</p>}
          <p className="mt-1 text-xs">No result is shown.</p>
        </div>
      )}

      {shown && single?.result_latex && (
        <div className="mt-3">
          <div className="text-xs uppercase tracking-wide text-muted">Result</div>
          <div className="overflow-x-auto">
            <Math latex={single.result_latex} display label="Result" />
          </div>
        </div>
      )}
      {shown && single?.status === "unchanged" && (
        <p className="mt-3 text-sm text-muted">{single.message}</p>
      )}

      {shown && stat && <StatBody result={stat} />}

      {shown && <Conditions items={conditions} title="Parameter space and conditions" />}
      {shown && <Steps steps={result.steps} />}

      {lines.length > 0 && (
        <div className="mt-4" data-testid="checks">
          <h3 className="text-sm font-medium">How it was checked</h3>
          <ul className="mt-1 space-y-1 text-sm">
            {lines.map((l) => (
              <li key={l.text} className={l.ok ? "" : "text-bad"}>
                {l.ok ? "✓" : "✗"} {l.text}
              </li>
            ))}
          </ul>
          <p className="mt-1 text-xs text-muted">Numerical agreement is evidence, not a proof.</p>
        </div>
      )}

      {shown &&
        result.message &&
        result.status === "ok" &&
        // for sums and expectations the engine's message only restates the check line above
        !(lines.length > 0 && (kind === "sums" || kind === "expectation")) && (
          <p className="mt-3 text-sm text-muted">{result.message}</p>
        )}

      {shown && continuations && continuations.length > 0 && onContinue && (
        <div className="mt-3 flex flex-wrap items-center gap-2 text-xs">
          <span className="text-muted">Continue from</span>
          {continuations.map((c, i) => (
            <button
              key={c.label}
              type="button"
              onClick={() => onContinue(i)}
              className="rounded border border-accent px-2 py-0.5 font-medium text-accent hover:bg-accent-soft"
            >
              {c.label}
            </button>
          ))}
        </div>
      )}
    </section>
  );
}

function StatBody({ result }: { result: StatModelResult }) {
  const moments = Object.entries(result.moments_latex ?? {});
  const expressions = Object.entries(result.expressions_latex ?? {});
  return (
    <>
      {result.estimates.length > 0 && (
        <div className="mt-3 space-y-3" data-testid="estimates">
          {result.estimates.map((e) => {
            const note = optimumText(e);
            return (
              <div key={e.parameter}>
                <div className="overflow-x-auto">
                  <Math
                    latex={`\\hat{${nameLatex(e.parameter)}} = ${e.latex}`}
                    display
                    label={`Estimate of ${e.parameter}`}
                  />
                </div>
                {note && <p className="text-sm">{note}</p>}
                {e.admissible_if.length > 0 && (
                  <Conditions items={e.admissible_if} title="The estimate exists only if" />
                )}
              </div>
            );
          })}
        </div>
      )}
      {moments.length > 0 && (
        <dl className="mt-3 space-y-1">
          {moments.map(([name, latex]) => (
            <div key={name} className="flex flex-wrap items-baseline gap-3 overflow-x-auto">
              <dt className="min-w-24 text-sm text-muted">
                <Math latex={MOMENT_LABEL[name] ?? name} />
              </dt>
              <dd>
                <Math latex={latex} />
              </dd>
            </div>
          ))}
        </dl>
      )}
      {expressions.length > 0 && (
        <div className="mt-3 space-y-2">
          {expressions.map(([name, latex]) => (
            <div key={name}>
              <div className="text-xs uppercase tracking-wide text-muted">
                {expressionLabel(name)}
              </div>
              <div className="overflow-x-auto">
                <Math latex={latex} display />
              </div>
            </div>
          ))}
        </div>
      )}
      {result.matrix_latex && (
        <div className="mt-3">
          <div className="text-xs uppercase tracking-wide text-muted">
            Fisher information matrix
          </div>
          <div className="overflow-x-auto">
            <Math latex={matrixLatex(result.matrix_latex)} display label="Information matrix" />
          </div>
        </div>
      )}
      {result.posterior && (
        <div className="mt-3" data-testid="posterior">
          <div className="text-xs uppercase tracking-wide text-muted">
            Posterior: {FAMILY_TITLE[result.posterior.family] ?? result.posterior.family}
          </div>
          <div className="overflow-x-auto">
            <Math
              latex={Object.entries(result.posterior.parameters_latex)
                .map(([n, l]) => `${nameLatex(n)}_{\\mathrm{post}} = ${l}`)
                .join(",\\quad ")}
              display
              label="Posterior parameters"
            />
          </div>
          <p className="text-xs text-muted">
            On the right, the parameter names are those of your prior and the data symbols are the
            ones you gave.
          </p>
          {result.posterior.dropped_factors_latex.length > 0 && (
            <p className="text-xs text-muted">
              Factors that do not depend on the parameter were dropped:{" "}
              {result.posterior.dropped_factors_latex.map((f, i) => (
                <span key={f}>
                  {i > 0 && ", "}
                  <Math latex={f} />
                </span>
              ))}
            </p>
          )}
        </div>
      )}
    </>
  );
}
