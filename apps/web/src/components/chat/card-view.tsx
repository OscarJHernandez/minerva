"use client";

import { Math } from "@minerva/ui/math";
import { useFormulaActions } from "./formula-actions";
import { ResultActions } from "../result-actions";
import { EquationNumber, RowNumbersProvider } from "./equation-number";
import type { Card } from "@/lib/chat-types";
import { DerivativePanel } from "../derivative-panel";
import { EntryCard } from "../entry-card";
import { IdentitiesPanel } from "../identities-panel";
import { SeriesPanel } from "../series-panel";
import { IntegralPanel } from "../integral-panel";
import { DistributionCard } from "./distribution-card";
import { StatMechCard } from "./statmech-card";
import { MatrixPanel } from "../matrix-panel";
import { VectorPanel } from "../vector-panel";
import { StatisticsPanel } from "../statistics-panel";
import { DimensionalCard } from "./dimensional-card";
import { DiscoveryCard } from "./discovery-card";
import { ApproximationCard } from "./approximation-card";
import { TelescopingCard } from "./telescoping-card";
import { GeometryCard } from "./geometry-card";
import { ExpansionCard } from "./expansion-card";
import { InvariantsCard } from "./invariants-card";
import { LagrangianCard } from "./lagrangian-card";
import { MotionCard } from "./motion-card";
import { ValuesNote } from "./values-note";
import { OdeAnalysisCard, OdeNumericCard, OdeSolutionCard } from "./ode-card";
import { PerturbationCard } from "./perturbation-card";
import { SolveCard } from "./solve-card";
import { IdentityCard } from "./identity-card";
import { UnitsCard } from "./units-card";
import { VariationCard } from "./variation-card";
import { ResultPanel, StatusPill } from "../result-panel";
import { IsolatePanel, StepResultPanel } from "../step-result-panel";
import type { Status } from "@/lib/types";
import { ProvenanceNotes, type Provenance } from "./provenance-notes";

/** Renders a tool result. The content comes from Minerva's own computation, never from the model. */
export function CardView({ card, streaming = false }: { card: Card; streaming?: boolean }) {
  // The agent loop attaches inherited restrictions and unverified-input provenance to every kind of
  // successful card, whatever its own result schema. Show them whether or not the model repeats them.
  return (
    <div className="min-w-0 space-y-2">
      <ProvenanceNotes data={(card.data ?? {}) as Provenance} />
      <CardResult card={card} streaming={streaming} />
    </div>
  );
}

function CardResult({ card, streaming = false }: { card: Card; streaming?: boolean }) {
  const { openInWorkbench } = useFormulaActions();
  const data = card.data as {
    status?: string;
    original_request?: unknown;
    result_text?: string;
    result_latex?: string;
  };
  const scalar =
    ["simplify", "differentiate", "integrate", "series"].includes(card.type) &&
    (data.status === "ok" || data.status === "unchanged") &&
    data.result_text;
  const actions = scalar && (
    <ResultActions
      text={data.result_text!}
      latex={data.result_latex}
      onOpen={
        !streaming && data.original_request && openInWorkbench
          ? () => openInWorkbench(card.id)
          : undefined
      }
    />
  );
  // The number and copy actions name the unrestricted derivative, even when the panel leads
  // with a separate conclusion on the equations of motion. Keep them with that exact formula.
  if (card.type === "differentiate") {
    return (
      <DerivativePanel
        result={card.data}
        total={card.tool === "time_derivative"}
        number={card.number}
        actions={actions}
      />
    );
  }
  const numbers = card.numbers && Object.keys(card.numbers).length > 0 ? card.numbers : null;
  const content = (
    <RowNumbersProvider value={numbers}>
      <CardContent card={card} />
    </RowNumbersProvider>
  );
  if (card.number == null)
    return (
      <>
        {content}
        {actions}
      </>
    );
  return (
    <div>
      <div className="flex justify-end">
        <EquationNumber n={card.number} />
      </div>
      {content}
      {actions}
    </div>
  );
}

function CardContent({ card }: { card: Card }) {
  switch (card.type) {
    case "simplify":
      return <ResultPanel result={card.data} />;
    case "identity":
      return <IdentityCard report={card.data} />;
    case "differentiate":
      return <DerivativePanel result={card.data} />;
    case "series":
      return <SeriesPanel result={card.data} />;
    case "integrate":
      return <IntegralPanel result={card.data} />;
    case "matrix":
      return <MatrixPanel result={card.data.result} methods={card.data.methods} />;
    case "sums":
      return (
        <StatisticsPanel kind="sums" result={card.data.result} headingId={`stat-${card.id}`} />
      );
    case "expectation":
      return (
        <StatisticsPanel
          kind="expectation"
          result={card.data.result}
          headingId={`stat-${card.id}`}
        />
      );
    case "statistics":
      return (
        <StatisticsPanel
          kind="statistics"
          result={card.data.result}
          headingId={`stat-${card.id}`}
        />
      );
    case "geometry":
      return <GeometryCard result={card.data.result} request={card.data.request} />;
    case "vector":
      return <VectorPanel result={card.data.result} request={card.data.request} />;
    case "distribution":
      return (
        <DistributionCard
          result={card.data.result}
          methods={card.data.methods}
          request={card.data.request}
        />
      );
    case "variation":
      return <VariationCard report={card.data.report} request={card.data.request} />;
    case "statmech":
      return (
        <StatMechCard result={card.data.result} args={card.data.args} answers={card.data.answers} />
      );
    case "identities":
      return <IdentitiesPanel result={card.data} />;
    case "step":
      return card.data.operation === "isolate" ? (
        <IsolatePanel result={card.data.result} />
      ) : (
        <StepResultPanel result={card.data.result} />
      );
    case "entries":
      if (card.data.entries.length === 0) {
        return (
          <p
            className="rounded-lg border border-rule bg-raised px-4 py-2 text-sm text-muted"
            aria-label={card.title}
          >
            Handbook search “{card.data.query}”: no entry matched. The handbook may not cover this.
          </p>
        );
      }
      return (
        <section className="rounded-lg border border-rule bg-raised p-4" aria-label={card.title}>
          <h3 className="font-serif text-lg">Handbook</h3>
          <p className="mb-3 text-xs text-muted">Search: “{card.data.query}”</p>
          <div className="grid gap-3 md:grid-cols-2">
            {card.data.entries.map((e) => (
              <EntryCard key={e.id} entry={e} />
            ))}
          </div>
        </section>
      );
    case "entry":
      return <EntryCard entry={card.data} />;
    case "ode_analysis":
      return <OdeAnalysisCard report={card.data.report} request={card.data.request} />;
    case "ode_solution":
      return <OdeSolutionCard report={card.data.report} request={card.data.request} />;
    case "ode_numeric":
      return (
        <>
          <ValuesNote chosenBy={card.data.values_chosen_by} values={card.data.assistant_values} />
          <OdeNumericCard report={card.data.report} request={card.data.request} />
        </>
      );
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
    case "telescoping":
      return <TelescopingCard result={card.data} />;
    case "approximation":
      return <ApproximationCard report={card.data.report} request={card.data.request} />;
    case "motion":
      return (
        <>
          <ValuesNote chosenBy={card.data.values_chosen_by} values={card.data.assistant_values} />
          <MotionCard report={card.data.report} request={card.data.request} />
        </>
      );
    case "dimensional":
      return <DimensionalCard report={card.data.report} methods={card.data.methods} />;
    case "units":
      return (
        <UnitsCard
          report={card.data.report}
          methods={card.data.methods}
          assumed={card.data.assumed_units ?? []}
        />
      );
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
    case "note":
      return <p className="text-xs text-muted">{card.data.message}</p>;
    case "error":
      return (
        <p role="alert" className="rounded bg-warn-soft px-3 py-2 text-sm text-warn">
          Tool could not run: {card.data.message}
        </p>
      );
  }
}
