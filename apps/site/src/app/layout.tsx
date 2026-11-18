import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Minerva — For the joy of figuring things out",
  description:
    "Your own open-source mathematical assistant for exploring ideas, discovering useful identities, and checking results. Self-host Minerva and bring your own model.",
  keywords: [
    "mathematical assistant",
    "math exploration",
    "open source",
    "self-hosted",
    "computer algebra",
    "symbolic computation",
    "SymPy",
    "mathematical identities",
    "tensor algebra",
    "Ollama",
    "local LLM",
  ],
  openGraph: {
    title: "Minerva — For the joy of figuring things out",
    description:
      "Explore mathematics for the joy of figuring things out. Open source, self-hostable, and grounded in verifiable symbolic computation.",
    type: "website",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-paper text-ink">{children}</body>
    </html>
  );
}
