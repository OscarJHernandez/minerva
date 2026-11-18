export const APP_URL = process.env.NEXT_PUBLIC_APP_URL ?? "http://localhost:3000";
import { explorePath } from "@minerva/ui/explorations";

/** The app's page for one exploration: the prompt of a landing example in the message box, not sent. */
export const exploreUrl = (id: string) => `${APP_URL}${explorePath(id)}`;

export const REPO_URL =
  process.env.NEXT_PUBLIC_REPO_URL ?? "https://github.com/OscarJHernandez/minerva";

/** A file in the repository's default branch. */
export const repoFile = (path: string) => `${REPO_URL}/blob/main/${path}`;

export const SELF_HOSTING_URL = repoFile("docs/self-hosting.md");
export const OPENROUTER_URL = repoFile("docs/openrouter-setup.md");
export const CONTRIBUTING_URL = repoFile("CONTRIBUTING.md");

export const ISSUES_URL = `${REPO_URL}/issues`;
