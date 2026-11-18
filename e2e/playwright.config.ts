import { defineConfig } from "@playwright/test";

/**
 * Browser tests of the landing page -> app flow, against the real API (deterministic scripted model, throwaway
 * SQLite database), the built web app and the built static site. Build first:
 *   NEXT_PUBLIC_APP_URL=http://localhost:3131 pnpm --filter @minerva/site build
 *   pnpm --filter @minerva/web build
 * Ports are unusual on purpose so that a developer's own servers on 3000/8100 are never reused by mistake.
 */
const API_PORT = 8131;
const WEB_PORT = 3131;
const SITE_PORT = 3132;

export default defineConfig({
  testDir: "./tests",
  timeout: 120_000,
  expect: { timeout: 30_000 },
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",
  use: { baseURL: `http://localhost:${WEB_PORT}`, trace: "retain-on-failure" },
  webServer: [
    {
      command: "bash e2e/start-api.sh",
      cwd: "..",
      url: `http://127.0.0.1:${API_PORT}/healthz`,
      reuseExistingServer: false,
      timeout: 120_000,
    },
    {
      command: `pnpm --filter @minerva/web exec next start --port ${WEB_PORT}`,
      cwd: "..",
      url: `http://localhost:${WEB_PORT}/login`,
      env: { MINERVA_API_URL: `http://127.0.0.1:${API_PORT}` },
      reuseExistingServer: false,
      timeout: 120_000,
    },
    {
      command: `python3 -m http.server ${SITE_PORT} --directory apps/site/out --bind 127.0.0.1`,
      cwd: "..",
      url: `http://127.0.0.1:${SITE_PORT}/`,
      reuseExistingServer: false,
      timeout: 60_000,
    },
  ],
});
