import type { Suite } from "@/lib/benchmarks";
import { repoFile } from "@/lib/links";

const PARTS = [
  { key: "correct", label: "answers computed and checked", tone: "bg-ok" },
  { key: "refused", label: "refused for a stated reason", tone: "bg-accent" },
  { key: "rejected", label: "wrong answers rejected", tone: "bg-warn" },
  { key: "pending", label: "known gaps", tone: "bg-bad" },
] as const;

/** One suite: what it checks, how its problems ended, and how to run it yourself. */
export function SuiteRow({ suite }: { suite: Suite }) {
  const parts = PARTS.filter((p) => suite[p.key] > 0);
  return (
    <article className="border-b border-rule py-6" aria-labelledby={`${suite.id}-t`}>
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <h3 id={`${suite.id}-t`} className="font-serif text-2xl tracking-tight">
          {suite.title}
        </h3>
        <p className="text-sm text-muted" data-testid={`suite-${suite.id}-total`}>
          {suite.total} problems
        </p>
      </div>
      <p className="mt-2 max-w-3xl text-muted">{suite.what}</p>
      <div
        className="mt-4 flex h-2.5 w-full overflow-hidden rounded-full bg-sunk"
        role="img"
        aria-label={parts.map((p) => `${suite[p.key]} ${p.label}`).join(", ")}
      >
        {parts.map((p) => (
          <div
            key={p.key}
            className={p.tone}
            style={{ width: `${(100 * suite[p.key]) / suite.total}%` }}
          />
        ))}
      </div>
      <ul className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-sm">
        {parts.map((p) => (
          <li key={p.key} className="flex items-center gap-2">
            <span aria-hidden="true" className={`size-2 rounded-full ${p.tone}`} />
            <span>
              <strong className="font-medium">{suite[p.key]}</strong>{" "}
              <span className="text-muted">{p.label}</span>
            </span>
          </li>
        ))}
      </ul>
      <p className="mt-3 text-sm text-muted">
        <a className="underline hover:text-ink" href={repoFile(suite.doc)}>
          How it is judged
        </a>
        {" · "}
        <a
          className="underline hover:text-ink"
          href={repoFile(`docs/benchmarks/${suite.sources[0]}`)}
        >
          Report
        </a>
        {" · "}
        <code className="break-words rounded bg-sunk px-1.5 py-0.5 text-xs">{suite.command}</code>
      </p>
    </article>
  );
}
