export const APP_URL = process.env.NEXT_PUBLIC_APP_URL ?? "http://localhost:3000";
export const REPO_URL =
  process.env.NEXT_PUBLIC_REPO_URL ?? "https://github.com/OscarJHernandez/minerva";

/** A file in the repository's default branch. */
export const repoFile = (path: string) => `${REPO_URL}/blob/main/${path}`;

export const SELF_HOSTING_URL = repoFile("docs/self-hosting.md");
export const CONTRIBUTING_URL = repoFile("CONTRIBUTING.md");
