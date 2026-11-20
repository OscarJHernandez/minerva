import { readFileSync } from "node:fs";
import { expect, test } from "@playwright/test";

/**
 * The Benchmarks page of the landing site: reachable from the header, and every number on it is the one in the
 * committed data file (generated from the reports by scripts/benchmarks_site.py), never a number of its own.
 */
const SITE = "http://localhost:3132";
const DATA = JSON.parse(
  readFileSync(new URL("../../apps/site/src/data/benchmarks.json", import.meta.url), "utf8"),
) as {
  suites: { id: string; total: number }[];
  chat: { rows: { id: string; model: { digest: string } }[] };
  models: { final: { model: { name: string; digest: string }; eligible: boolean }[] };
};

test("the Benchmarks tab opens the page, and the page shows the committed numbers", async ({
  page,
}) => {
  await page.goto(SITE);
  await page
    .getByRole("navigation", { name: "Main" })
    .getByRole("link", { name: "Benchmarks" })
    .click();
  await expect(page).toHaveURL(/\/benchmarks\/?$/);
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("How Minerva is measured");
  const total = DATA.suites.reduce((n, s) => n + s.total, 0);
  await expect(page.getByTestId("problem-total")).toHaveText(String(total));
  for (const s of DATA.suites) {
    await expect(page.getByTestId(`suite-${s.id}-total`)).toHaveText(`${s.total} problems`);
  }
  // A model-dependent number is never shown without the model as its host reported it.
  const digest = DATA.chat.rows.find((r) => r.id === "all")!.model.digest;
  await expect(page.getByTestId("chat-model")).toContainText(`digest ${digest}`);
  await expect(page.getByTestId("chat-model")).toContainText(/thinking (on|off)/);
  await expect(page.getByTestId("false-accepts")).toHaveText("0");

  // The model comparison names every model by what it is, with its digest, and shows a false-verified run as such.
  for (const row of DATA.models.final) {
    const card = page.getByTestId(`model-${row.model.digest}`);
    await expect(card).toContainText(row.model.name);
    await expect(card).toContainText(`digest ${row.model.digest}`);
    await expect(card.getByTestId("not-eligible")).toHaveCount(row.eligible ? 0 : 1);
  }
});
