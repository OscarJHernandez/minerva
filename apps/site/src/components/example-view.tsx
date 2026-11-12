import { Math } from "@minerva/ui/math";
import { statusTone, type Example } from "@/lib/capabilities";

export function StatusBadge({ status, label }: { status: string; label?: string }) {
  const ok = statusTone(status) === "ok";
  return (
    <span
      className={`shrink-0 rounded-full px-2.5 py-0.5 text-xs font-medium ${
        ok ? "bg-ok-soft text-ok" : "bg-warn-soft text-warn"
      }`}
    >
      {label ?? status.replaceAll("_", " ")}
    </span>
  );
}

/** One scrollable line of math, so a long formula never widens the page on a phone. */
export function MathLine({ latex }: { latex: string }) {
  return (
    <div className="overflow-x-auto overflow-y-hidden py-0.5 text-lg">
      <Math latex={latex} />
    </div>
  );
}

function Labelled({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <dt className="text-xs uppercase tracking-wide text-muted">{label}</dt>
      <dd className="mt-0.5">{children}</dd>
    </div>
  );
}

/** A recorded example: the input, what the engine returned, its status and verification. */
export function ExampleView({ ex }: { ex: Example }) {
  return (
    <article className="min-w-0 rounded-lg border border-rule bg-raised p-5 sm:p-6">
      <div className="flex items-baseline justify-between gap-3">
        <h4 className="font-serif text-lg">{ex.title}</h4>
        <StatusBadge status={ex.status} />
      </div>
      <dl className="mt-4 space-y-3">
        {ex.input.map((line) => (
          <Labelled key={line.label} label={line.label}>
            {line.source ? (
              <code className="block overflow-x-auto whitespace-pre text-sm">{line.source}</code>
            ) : (
              <span className="text-sm">{line.text}</span>
            )}
          </Labelled>
        ))}
        {ex.assumptions && (
          <Labelled label="Assumptions">
            <span className="text-sm">{ex.assumptions}</span>
          </Labelled>
        )}
        {ex.outputs.map((out) => (
          <Labelled key={`${out.label}-${out.latex ?? out.text}`} label={out.label}>
            {out.latex && <MathLine latex={out.latex} />}
            {out.text && <p className="text-sm text-muted">{out.text}</p>}
          </Labelled>
        ))}
      </dl>
      {(ex.checks.length > 0 || ex.notes.length > 0) && (
        <ul className="mt-4 space-y-1 border-t border-rule pt-3 text-sm text-muted">
          {ex.checks.map((c) => (
            <li key={c}>
              <span className="font-medium text-ink">Check:</span> {c}
            </li>
          ))}
          {ex.notes.map((n) => (
            <li key={n}>{n}</li>
          ))}
        </ul>
      )}
    </article>
  );
}
