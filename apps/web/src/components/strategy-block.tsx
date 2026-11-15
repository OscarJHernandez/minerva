import { CopyMath as Formula } from "@/components/copyable-math";
import type { IntegrationTechnique, StrategyReport } from "@/lib/types";

const NAME: Record<IntegrationTechnique, string> = {
  direct: "SymPy's direct route",
  substitution: "Substitution",
  parts: "Integration by parts",
  trigonometric: "Trigonometric identities",
  partial_fractions: "Partial fractions",
  parity: "Symmetry (parity)",
  reflection: "Reflection (King's rule)",
  frullani: "Frullani integral",
  schwinger: "Schwinger parametrization",
  parameter_differentiation: "Differentiation under the integral sign",
  numerical: "Numerical estimate",
};

const WHY: Record<string, string> = {
  composite_with_inner_derivative:
    "a function of an inner expression, possibly with its derivative",
  polynomial_times_transcendental:
    "a polynomial times an exponential, logarithm or trigonometric function",
  trigonometric_powers_or_products: "trigonometric powers or products",
  rational_with_reducible_denominator: "a rational function with a reducible denominator",
  odd_on_symmetric_interval: "odd integrand on a symmetric interval",
  even_on_symmetric_interval: "even integrand on a symmetric interval",
  reflection_simplifies: "x ↦ a + b − x makes the average simpler",
  difference_at_two_scalings_over_x: "a difference of one function at two scalings, over x",
  negative_power_of_a_positive_base: "a negative power of a positive expression",
  free_parameter: "the integrand depends on a parameter",
  no_closed_form_for_a_number: "a definite integral with no closed form found",
};

const VERIFICATION: Record<StrategyReport["verification"], string> = {
  proved: "Verified independently of the technique: numeric derivative check and a symbolic zero",
  numerically_checked: "Verified independently of the technique by numerical checks (not a proof)",
  numeric_estimate: "A numerical estimate from two independent rules: not a formula, not a proof",
  none: "",
};

/**
 * The technique behind an integral: why it applies, its steps, what else was suggested and what was
 * tried. A technique's own steps are shown only for a result that passed the independent checks.
 */
export function StrategyBlock({ strategy }: { strategy: StrategyReport }) {
  const selected =
    strategy.technique && strategy.technique !== "direct" ? strategy.technique : null;
  const blocked = strategy.attempts.filter(
    (a) => a.outcome === "blocked" || a.outcome === "failed_verification",
  );
  if (!selected && strategy.suggestions.length === 0 && blocked.length === 0) return null;
  return (
    <div className="mt-4 rounded border border-rule p-3" aria-label="Method">
      <div className="text-xs uppercase tracking-wide text-muted">Method</div>
      {selected && (
        <>
          <p className="mt-1 text-sm font-medium">
            {NAME[selected]}
            {strategy.why && (
              <span className="font-normal text-muted"> — {WHY[strategy.why] ?? strategy.why}</span>
            )}
          </p>
          <ol className="mt-1 space-y-1 text-sm">
            {strategy.steps.map((step, i) => (
              <li key={i}>
                <span className="text-muted">{step.what}</span>
                <Formula latex={step.latex} display />
              </li>
            ))}
          </ol>
          {strategy.conditions_latex.length > 0 && (
            <ul className="mt-1 text-sm text-muted">
              {strategy.conditions_latex.map((c) => (
                <li key={c}>
                  established: <Formula latex={c} />
                </li>
              ))}
            </ul>
          )}
          {VERIFICATION[strategy.verification] && (
            <p className="mt-1 text-sm text-muted">{VERIFICATION[strategy.verification]}</p>
          )}
        </>
      )}
      {!selected && strategy.suggestions.length > 0 && (
        <p className="mt-1 text-sm">
          Techniques that may apply: {strategy.suggestions.map((s) => NAME[s.technique]).join(", ")}
          .
        </p>
      )}
      {blocked.map((a) => (
        <p key={a.technique} className="mt-1 text-sm text-warn">
          {NAME[a.technique]}:{" "}
          {a.outcome === "blocked"
            ? `not applied, a condition is not established (${a.detail.replaceAll("_", " ")})`
            : "a candidate failed verification and is not shown"}
        </p>
      ))}
      {strategy.note && <p className="mt-1 text-sm text-warn">{strategy.note}</p>}
    </div>
  );
}
