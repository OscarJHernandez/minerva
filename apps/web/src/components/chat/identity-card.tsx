import { CopyMath as Math } from "@/components/copyable-math";
import type { IdentityReport } from "@/lib/chat-types";
import { StatusPill } from "../result-panel";

const VERDICT: Record<NonNullable<IdentityReport["verdict"]>, { label: string; tone: string }> = {
  proved_equal: { label: "Equal: proved", tone: "bg-ok-soft text-ok" },
  numerically_equal: {
    label: "Equal: checked numerically, not a proof",
    tone: "bg-ok-soft text-ok",
  },
  conditional: { label: "Not an identity as stated", tone: "bg-warn-soft text-warn" },
  not_identity: { label: "Not an identity", tone: "bg-warn-soft text-warn" },
};

/** The verdict on one stated equation, and the exact point where its sides differ when it is not an identity. */
export function IdentityCard({ report }: { report: IdentityReport }) {
  const verdict = report.verdict ? VERDICT[report.verdict] : null;
  const cx = report.counterexample;
  return (
    <section className="rounded-lg border border-rule bg-raised p-5" aria-label="Identity check">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="font-serif text-xl">Is it an identity?</h3>
        {verdict ? (
          <span
            className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-medium ${verdict.tone}`}
          >
            {verdict.label}
          </span>
        ) : (
          <StatusPill status={report.status} />
        )}
      </div>
      {report.lhs_latex && report.rhs_latex && (
        <Math latex={`${report.lhs_latex} = ${report.rhs_latex}`} display />
      )}
      <p className="text-sm text-muted">{report.message}</p>
      {report.verdict === "conditional" && report.condition && (
        <p className="mt-2 text-sm" data-testid="identity-condition">
          {report.condition}
        </p>
      )}
      {cx && (
        <div className="mt-3 rounded border border-rule p-3 text-sm" data-testid="identity-witness">
          <div className="font-medium">A point where the two sides differ</div>
          <ul className="mt-1 space-y-1">
            {Object.entries(cx.substitution).map(([name, value]) => (
              <li key={name}>
                <Math latex={`${name} = ${value}`} />
              </li>
            ))}
          </ul>
          <p className="mt-1">
            <span className="text-muted">left side</span> <Math latex={cx.lhs_value} />{" "}
            <span className="text-muted">right side</span> <Math latex={cx.rhs_value} />
          </p>
          <p className="mt-1 text-xs text-muted">
            An exact point, checked at 60 and 120 digits (a {cx.probe} probe).
          </p>
        </div>
      )}
    </section>
  );
}
