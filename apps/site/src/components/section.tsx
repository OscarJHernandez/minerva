export function Section({
  id,
  title,
  children,
}: {
  id: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section id={id} aria-labelledby={`${id}-h`} className="border-t border-rule py-16 sm:py-20">
      <div className="mx-auto max-w-5xl px-5 sm:px-8">
        <h2 id={`${id}-h`} className="font-serif text-3xl tracking-tight sm:text-4xl">
          {title}
        </h2>
        <div className="mt-8">{children}</div>
      </div>
    </section>
  );
}
