import { readFileSync } from "node:fs";
import { expect, test, type Page } from "@playwright/test";

/**
 * "Explore in Minerva": a verified landing-page example opens the real chat with its mathematical context and the
 * user runs it. Everything that is a result below comes from the engine through the API (the model here is the
 * deterministic scripted stand-in, which only chooses the tool); nothing is read from the page's own data.
 */
const SITE = "http://localhost:3132";
const REGISTRY = JSON.parse(
  readFileSync(new URL("../../packages/ui/fixtures/explorations.json", import.meta.url), "utf8"),
) as {
  explorations: { id: string; prompt: string; button: string; approved: boolean }[];
};
const FULL = REGISTRY.explorations.find((e) => e.id === "schwarzschild-curvature")!;

async function register(page: Page) {
  const email = `explorer-${Date.now()}-${Math.floor(Math.random() * 1e6)}@example.com`;
  await page.getByLabel("Email").fill(email);
  await page.getByLabel("Password").fill("a-long-test-password-1");
  await page.getByRole("button", { name: "Create account" }).click();
}

test("the landing page shows an Explore button only for an approved exploration", async ({
  page,
}) => {
  await page.goto(SITE);
  await expect(
    page.getByRole("link", { name: "Inspect the curvature calculation and assumptions" }),
  ).toHaveAttribute("href", "/docs#differential-geometry");
  for (const e of REGISTRY.explorations) {
    const link = page.locator(`a[href$="/explore/${e.id}"]`);
    if (e.approved) {
      await expect(link).toHaveCount(1);
      await expect(link).toHaveText(e.button);
    } else {
      await expect(link).toHaveCount(0);
    }
  }
  if (FULL.approved) {
    await page.locator(`a[href$="/explore/${FULL.id}"]`).click();
    await expect(page).toHaveURL(
      /localhost:3131\/login\?next=%2Fexplore%2Fschwarzschild-curvature/,
    );
  }
});

test("an exploration survives login, fills the message box without sending, runs on the engine and can be continued", async ({
  page,
}) => {
  // A logged-out visitor is sent to log in and comes back to the exploration, also through "Create an account".
  await page.goto(`/explore/${FULL.id}`);
  await expect(page).toHaveURL(/\/login\?next=%2Fexplore%2Fschwarzschild-curvature/);
  await page.getByRole("link", { name: "Create an account" }).click();
  await expect(page).toHaveURL(/\/register\?next=%2Fexplore%2Fschwarzschild-curvature/);
  await register(page);
  await expect(page).toHaveURL(/\/explore\/schwarzschild-curvature$/);

  // The original context is in the box, exactly, and nothing was sent.
  const box = page.getByLabel("Message");
  await expect(box).toHaveValue(FULL.prompt);
  await expect(page.getByTestId("quantity-ricci")).toHaveCount(0);

  // The user runs it: the engine computes the quantities and the evidence kinds stay apart.
  await page.getByRole("button", { name: "Send", exact: true }).click();
  const ricci = page.getByTestId("quantity-ricci");
  await expect(ricci).toContainText("Every component is zero.");
  await expect(ricci).toContainText("Proved by a symbolic zero test, and numerically checked");
  const kretschmann = page.getByTestId("quantity-kretschmann");
  await expect(kretschmann).toContainText("Numerically checked; no symbolic proof");
  await expect(kretschmann).not.toContainText("Proved by a symbolic zero test");
  await expect(page.getByTestId("quantity-christoffel")).toBeVisible();
  await expect(page.getByTestId("restrictions")).toBeVisible();
  await expect(page.getByTestId("signature")).toContainText("NOT DETERMINED", {
    ignoreCase: true,
  });
  await expect(box).toHaveValue("");

  // The conversation has its own address now; the metric is reused for another quantity from the card.
  await expect(page).toHaveURL(/\/c\/[0-9a-f-]{36}$/);
  await page.getByRole("button", { name: "Einstein tensor" }).click();
  await expect(page.getByTestId("quantity-einstein")).toContainText("Every component is zero.");

  // The user keeps talking in the same conversation, and a reload keeps everything.
  await box.fill("Now compute the Ricci scalar of the Schwarzschild metric");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect(page.getByTestId("quantity-ricci_scalar")).toBeVisible();
  await page.reload();
  await expect(page.getByTestId("quantity-ricci_scalar")).toBeVisible();
  await expect(page.getByText("Using the Schwarzschild metric").first()).toBeVisible();
  await expect(page.getByTestId("quantity-kretschmann").first()).toContainText(
    "Numerically checked; no symbolic proof",
  );
});

test("an unknown exploration is not found", async ({ page }) => {
  await page.goto("/register");
  await register(page);
  await expect(page.getByRole("button", { name: "Log out" })).toBeVisible();
  const response = await page.goto("/explore/not-an-exploration");
  expect(response?.status()).toBe(404);
});
