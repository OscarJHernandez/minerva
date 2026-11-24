import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Card } from "@/lib/chat-types";
import fixture from "../../../../../packages/ui/fixtures/equation-rows.json";
import { MatrixPanel } from "../matrix-panel";
import { OdeAnalysisPanel } from "../ode-analysis-panel";
import { OdeSolutionPanel } from "../ode-solution-panel";
import { StatisticsPanel } from "../statistics-panel";
import { StatMechPanel, quantityKey, quantitySymbol } from "../statmech-panel";
import { DimensionalCard } from "./dimensional-card";
import { PerturbationCard } from "./perturbation-card";
import { CardView } from "./card-view";
import { RowNumbersProvider } from "./equation-number";

/**
 * The keys the server numbers (`services/api/.../card_equations.py`) are pinned by the same fixture in
 * `test_equation_rows_contract.py`. Here every key of the fixture is given a distinct number and the panels
 * must show exactly those numbers: a row the server numbers and the panel does not mark (or the reverse)
 * fails one of the two tests.
 */
type Entry = { data: { report: unknown; methods?: unknown[] }; keys: string[] };
const entries = fixture as unknown as Record<"solve" | "ode_solution" | "ode_analysis", Entry>;

function numbersFor(keys: string[]): Record<string, number> {
  return Object.fromEntries(keys.map((k, i) => [k, 100 + i]));
}

function shown(): string[] {
  return screen.queryAllByTestId("equation-number").map((e) => e.textContent ?? "");
}

function card(kind: "solve" | "ode_solution" | "ode_analysis", numbers: Record<string, number>) {
  const entry = entries[kind];
  return {
    id: "c",
    tool: kind,
    type: kind,
    title: kind,
    data: { ...entry.data, methods: [], request: {} },
    numbers,
  } as unknown as Card;
}

describe("row numbers of the solve and differential equation cards", () => {
  it.each(["solve", "ode_solution", "ode_analysis"] as const)(
    "%s: every key the server numbers is a marked row, and nothing else is",
    (kind) => {
      const keys = entries[kind].keys;
      const numbers = numbersFor(keys);
      render(<CardView card={card(kind, numbers)} />);
      const expected = new Set(Object.values(numbers).map((n) => `(${n})`));
      expect(new Set(shown())).toEqual(expected);
      expect(shown()).toHaveLength(keys.length);
    },
  );

  it.each(["solve", "ode_solution", "ode_analysis"] as const)(
    "%s: a key the server did not number shows no number",
    (kind) => {
      const keys = entries[kind].keys;
      const numbers = numbersFor(keys);
      delete numbers[keys[0] ?? ""];
      render(<CardView card={card(kind, numbers)} />);
      expect(shown()).toHaveLength(keys.length - 1);
    },
  );

  it("shows no number without any, as in a card from before they existed", () => {
    render(<CardView card={card("solve", {})} />);
    expect(screen.queryByTestId("equation-number")).not.toBeInTheDocument();
  });

  it("the workbench panels render unchanged without a provider", () => {
    const solution = entries.ode_solution.data.report as Parameters<
      typeof OdeSolutionPanel
    >[0]["report"];
    const analysis = entries.ode_analysis.data.report as Parameters<
      typeof OdeAnalysisPanel
    >[0]["report"];
    render(
      <>
        <OdeSolutionPanel report={solution} />
        <OdeAnalysisPanel report={analysis} />
      </>,
    );
    expect(screen.queryByTestId("equation-number")).not.toBeInTheDocument();
    expect(document.body.textContent).toContain("Closed form");
  });
});

/**
 * Variation, statistics and expansion. Each entry also carries `rows`: the LaTeX the server numbers under each
 * key. Every key is given a distinct number and the number must sit next to exactly that formula (read from
 * KaTeX's own TeX annotation), so a panel that numbers another formula under the right key fails here.
 */
type RowEntry = {
  data: Record<string, unknown>;
  keys: string[];
  rows: Record<string, string>;
};
const rowEntries = fixture as unknown as Record<string, RowEntry>;
const NEW_ENTRIES: Record<
  string,
  | "variation"
  | "statistics"
  | "expansion"
  | "statmech"
  | "perturbation"
  | "dimensional"
  | "ode_analysis"
  | "matrix"
> = {
  variation: "variation",
  variation_second: "variation",
  statistics_moments: "statistics",
  statistics_estimate: "statistics",
  statistics_expressions: "statistics",
  statistics_fisher: "statistics",
  expansion: "expansion",
  expansion_two_modes: "expansion",
  statmech: "statmech",
  statmech_grand: "statmech",
  statmech_micro: "statmech",
  statmech_variance_N: "statmech",
  statmech_mean_E: "statmech",
  perturbation: "perturbation",
  dimensional: "dimensional",
  ode_equilibria: "ode_analysis",
  matrix_gradient: "matrix",
  matrix_integral: "matrix",
};

function rowCard(name: string, numbers: Record<string, number>) {
  const entry = rowEntries[name];
  const type = NEW_ENTRIES[name];
  return {
    id: "c",
    tool: type,
    type,
    title: type,
    data: { methods: [], request: {}, ...entry?.data },
    numbers,
  } as unknown as Card;
}

function texNextTo(n: number): string[] {
  return screen
    .queryAllByTestId("equation-number")
    .filter((e) => e.textContent === `(${n})`)
    .map(
      (e) =>
        e.parentElement?.querySelector('annotation[encoding="application/x-tex"]')?.textContent ??
        "",
    );
}

describe("row numbers of the variation, statistics and expansion cards", () => {
  it.each(Object.keys(NEW_ENTRIES))(
    "%s: each key the server numbers marks exactly the formula the server numbered",
    (name) => {
      const { keys, rows } = rowEntries[name] as RowEntry;
      const numbers = numbersFor(keys);
      render(<CardView card={rowCard(name, numbers)} />);
      expect(shown()).toHaveLength(keys.length);
      for (const key of keys) {
        expect(texNextTo(numbers[key] as number), key).toEqual([rows[key]]);
      }
    },
  );

  it.each(Object.keys(NEW_ENTRIES))(
    "%s: a key the server did not number shows no number",
    (name) => {
      const { keys } = rowEntries[name] as RowEntry;
      const numbers = numbersFor(keys);
      delete numbers[keys[0] ?? ""];
      render(<CardView card={rowCard(name, numbers)} />);
      expect(shown()).toHaveLength(keys.length - 1);
    },
  );

  it.each(Object.keys(NEW_ENTRIES))("%s: no number without any, as in an older card", (name) => {
    render(<CardView card={rowCard(name, {})} />);
    expect(screen.queryByTestId("equation-number")).not.toBeInTheDocument();
  });

  it("the shared statistics panel renders unchanged without a provider", () => {
    const moments = rowEntries.statistics_moments as RowEntry;
    render(
      <StatisticsPanel
        kind="statistics"
        result={moments.data.result as Parameters<typeof StatisticsPanel>[0]["result"]}
        headingId="s"
      />,
    );
    expect(screen.queryByTestId("equation-number")).not.toBeInTheDocument();
    expect(document.body.textContent).toContain("Moments");
  });

  it("the statmech, perturbation and dimensional panels render unchanged without a provider", () => {
    const stat = rowEntries.statmech as RowEntry;
    const pert = rowEntries.perturbation as RowEntry;
    const dim = rowEntries.dimensional as RowEntry;
    render(
      <>
        <StatMechPanel
          result={stat.data.result as Parameters<typeof StatMechPanel>[0]["result"]}
          needs={{ cell: false, identical: false }}
        />
        <PerturbationCard
          report={pert.data.report as Parameters<typeof PerturbationCard>[0]["report"]}
        />
        <DimensionalCard
          report={dim.data.report as Parameters<typeof DimensionalCard>[0]["report"]}
          methods={[]}
        />
      </>,
    );
    expect(screen.queryByTestId("equation-number")).not.toBeInTheDocument();
    expect(document.body.textContent).toContain("Dimensionless groups");
    expect(document.body.textContent).toContain("Perturbation solution");
  });

  it("the panel draws the quantities outside the old table as LaTeX symbols, and a user label as named", () => {
    const base = { variable: null };
    expect(quantitySymbol({ name: "Xi", ...base })).toBe("\\Xi");
    expect(quantitySymbol({ name: "lnXi", ...base })).toBe("\\ln \\Xi");
    expect(quantitySymbol({ name: "Omega", ...base })).toBe("\\Omega");
    expect(quantitySymbol({ name: "N", ...base })).toBe("\\langle N \\rangle");
    expect(quantitySymbol({ name: "variance_N", ...base })).toBe("\\operatorname{Var}(N)");
    // a name the table does not have, and a label the user chose that equals a table name, stay as named
    expect(quantitySymbol({ name: "occupations", ...base })).toBe("occupations");
    expect(quantitySymbol({ name: "S", variable: "beta" })).toBe("S");
  });

  it("the occupation list is drawn and numbered per state, and an unsplittable one as before", () => {
    const grand = (rowEntries.statmech_grand as RowEntry).data.result as Parameters<
      typeof StatMechPanel
    >[0]["result"];
    const result = (latex: string) => ({
      ...grand,
      steps: [],
      quantities: [
        {
          ...grand.quantities[0]!,
          name: "occupations",
          variable: null,
          checks: [],
          latex,
        },
      ],
    });
    const two = String.raw`\langle n\rangle_{1} = a,\quad \langle n\rangle_{2} = b`;
    const panel = (latex: string) => (
      <RowNumbersProvider value={{ "occupation:1": 4, "occupation:2": 5 }}>
        <StatMechPanel result={result(latex)} needs={{ cell: false, identical: false }} />
      </RowNumbersProvider>
    );
    const { unmount } = render(panel(two));
    expect(screen.queryAllByTestId("equation-number").map((e) => e.textContent)).toEqual([
      "(4)",
      "(5)",
    ]);
    unmount();
    // a text that does not split exactly shows no number, whatever the provider holds
    render(panel(String.raw`n_{1} = a,\quad n_{2} = b`));
    expect(screen.queryByTestId("equation-number")).not.toBeInTheDocument();
  });

  it("a repeated quantity name gets a distinct key, as on the server", () => {
    const names = ["Z", "U", "Z", "Z"].map((name) => ({ name }));
    expect(names.map((_, i) => quantityKey(names, i))).toEqual([
      "quantity:Z",
      "quantity:U",
      "quantity:Z:2",
      "quantity:Z:3",
    ]);
  });

  it("a sums card shows the number of its whole result", () => {
    const card = {
      id: "s",
      tool: "sums",
      type: "sums",
      title: "sums",
      data: {
        result: {
          status: "ok",
          action: "expand",
          result_latex: "n x",
          steps: [],
          conditions: [],
          message: "",
        },
        request: {},
      },
      number: 7,
    } as unknown as Card;
    render(<CardView card={card} />);
    expect(shown()).toEqual(["(7)"]);
  });
});

describe("row numbers of the equilibria of an ODE analysis and the whole number of an approximation", () => {
  type Equilibria = { report: { qualitative: { equilibria: { eigenvalues_latex: string[] }[] } } };

  it("the rows are numbered by position in the canonical order, whatever order the solver gave", () => {
    const { keys, rows } = rowEntries.ode_equilibria as RowEntry;
    const numbers = numbersFor(keys);
    const reference = rowCard("ode_equilibria", numbers);
    const shuffled = structuredClone(reference) as Card;
    const q = (shuffled.data as unknown as Equilibria).report.qualitative;
    q.equilibria.reverse();
    for (const e of q.equilibria) e.eigenvalues_latex.reverse();
    render(<CardView card={shuffled} />);
    expect(shown()).toHaveLength(keys.length);
    for (const key of keys.filter((k) => k.startsWith("equil:"))) {
      expect(texNextTo(numbers[key] as number), key).toEqual([rows[key]]);
    }
  });

  it("an approximation shows the number of its whole result", () => {
    const entry = fixture as unknown as Record<string, { data: Record<string, unknown> }>;
    const card = {
      id: "a",
      tool: "approximation",
      type: "approximation",
      title: "Approximation",
      data: { ...entry.approximation?.data },
      number: 9,
    } as unknown as Card;
    const { unmount } = render(<CardView card={card} />);
    expect(shown()).toEqual(["(9)"]);
    unmount();
    render(<CardView card={{ ...card, number: undefined }} />);
    expect(screen.queryByTestId("equation-number")).not.toBeInTheDocument();
  });
});

describe("row numbers of the matrix card: steps and alternative conventions", () => {
  it("the whole number of the card and the row numbers sit side by side", () => {
    const { keys, rows } = rowEntries.matrix_gradient as RowEntry;
    const numbers = numbersFor(keys);
    const card = { ...rowCard("matrix_gradient", numbers), number: 99 } as unknown as Card;
    render(<CardView card={card} />);
    expect(shown()).toHaveLength(keys.length + 1);
    expect(texNextTo(numbers["alt:0"] as number)).toEqual([rows["alt:0"]]);
  });

  it("the workbench renders the matrix panel unchanged without a provider", () => {
    const entry = rowEntries.matrix_gradient as RowEntry;
    render(
      <MatrixPanel result={entry.data.result as Parameters<typeof MatrixPanel>[0]["result"]} />,
    );
    expect(screen.queryByTestId("equation-number")).not.toBeInTheDocument();
    expect(document.body.textContent).toContain("Steps");
    expect(document.body.textContent).toContain("Entrywise form");
  });
});
