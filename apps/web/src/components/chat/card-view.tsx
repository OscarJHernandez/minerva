import { Math } from "@minerva/ui/math";
import type { Card } from "@/lib/chat-types";
import { DerivativePanel } from "../derivative-panel";
import { EntryCard } from "../entry-card";
import { IdentitiesPanel } from "../identities-panel";
import { SeriesPanel } from "../series-panel";
import { IntegralPanel } from "../integral-panel";
import { MatrixPanel } from "../matrix-panel";
import { DimensionalCard } from "./dimensional-card";
import { DiscoveryCard } from "./discovery-card";
import { ExpansionCard } from "./expansion-card";
import { InvariantsCard } from "./invariants-card";
import { LagrangianCard } from "./lagrangian-card";
import { MotionCard } from "./motion-card";
import { PerturbationCard } from "./perturbation-card";
import { SolveCard } from "./solve-card";
import { UnitsCard } from "./units-card";
import { ResultPanel, StatusPill } from "../result-panel";
import type { Status } from "@/lib/types";

/** Renders a tool result. The content comes from Minerva's own computation, never from the model. */
export function CardView({ card }: { card: Card }) {
  switch (card.type) {
    case "simplify":
      return <ResultPanel result={card.data} />;
    case "differentiate":
      return <DerivativePanel result={card.data} />;
    case "series":
      return <SeriesPanel result={card.data} />;
    case "integrate":
      return <IntegralPanel result={card.data} />;
    case "matrix":
      return <MatrixPanel result={card.data.result} methods={card.data.methods} />;
    case "identities":
      return <IdentitiesPanel result={card.data} />;
    case "entries":
      return (
        <section className="rounded-lg border border-rule bg-raised p-4" aria-label={card.title}>
          <h3 className="font-serif text-lg">Handbook</h3>
          <p className="mb-3 text-xs text-muted">Search: “{card.data.query}”</p>
          {card.data.entries.length === 0 ? (
            <p className="text-sm text-muted">No entries matched.</p>
          ) : (
            <div className="grid gap-3 md:grid-cols-2">
              {card.data.entries.map((e) => (
                <EntryCard key={e.id} entry={e} />
              ))}
            </div>
          )}
        </section>
      );
    case "entry":
      return <EntryCard entry={card.data} />;
    case "solve":
      return <SolveCard report={card.data.report} methods={card.data.methods} />;
    case "lagrangian":
      return (
        <LagrangianCard
          report={card.data.report}
          methods={card.data.methods}
          request={card.data.request}
        />
      );
    case "expansion":
      return <ExpansionCard report={card.data.report} />;
    case "perturbation":
      return <PerturbationCard report={card.data.report} request={card.data.request} />;
    case "invariants":
      return <InvariantsCard report={card.data.report} />;
    case "discovery":
      return <DiscoveryCard report={card.data.report} />;
    case "motion":
      return <MotionCard report={card.data.report} request={card.data.request} />;
    case "dimensional":
      return <DimensionalCard report={card.data.report} methods={card.data.methods} />;
    case "units":
      return <UnitsCard report={card.data.report} methods={card.data.methods} />;
    case "tensor":
      return (
        <section className="rounded-lg border border-rule bg-raised p-4" aria-label={card.title}>
          <div className="flex items-center justify-between">
            <h3 className="font-serif text-lg">Tensor canonical form</h3>
            <StatusPill status={card.data.status as Status} />
          </div>
          {card.data.input_text && (
            <p className="mt-2 font-mono text-xs text-muted">{card.data.input_text}</p>
          )}
          {card.data.result_latex && <Math latex={card.data.result_latex} display />}
          <p className="text-sm text-muted">{card.data.message}</p>
        </section>
      );
    case "error":
      return (
        <p role="alert" className="rounded bg-warn-soft px-3 py-2 text-sm text-warn">
          Tool could not run: {card.data.message}
        </p>
      );
  }
}
