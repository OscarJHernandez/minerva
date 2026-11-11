import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Minerva — a mathematical workbench",
  description:
    "Open-source computer algebra with a citable handbook of identities, tables and physical equations. Every result is verified; every identity states what it assumes.",
  openGraph: {
    title: "Minerva — a mathematical workbench",
    description: "Verified symbolic computation and a citable handbook of mathematics and physics.",
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
