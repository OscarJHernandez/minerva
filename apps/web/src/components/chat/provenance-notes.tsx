import { Math } from "@minerva/ui/math";
import provenance from "../../../../../packages/ui/fixtures/input-provenance.json";

/** Ends an entry of `unverified_inputs` that names one row of an earlier card's list (see refs.py). */
export const COMPONENT_SUFFIX: string = provenance.component_suffix;

/** Ends an entry that names a tool result with an equation number but no result id (see refs.py). */
export const NO_ID_SUFFIX: string = provenance.no_result_id_suffix;

export interface Provenance {
  inherited_restrictions_latex?: string[] | null;
  unverified_inputs?: string[] | null;
}

/**
 * Qualifications the agent loop stored on a successful card. They are displayed exactly as stored:
 * nothing here decides a condition, links an input or upgrades a verification. Old cards lack both fields.
 */
export function ProvenanceNotes({ data }: { data: Provenance }) {
  const restrictions = data.inherited_restrictions_latex ?? [];
  const inputs = data.unverified_inputs ?? [];
  const written = inputs.filter((s) => !s.endsWith(COMPONENT_SUFFIX) && !s.endsWith(NO_ID_SUFFIX));
  const parts = inputs.filter((s) => s.endsWith(COMPONENT_SUFFIX));
  const unnumbered = inputs.filter((s) => s.endsWith(NO_ID_SUFFIX));
  if (
    restrictions.length === 0 &&
    written.length === 0 &&
    parts.length === 0 &&
    unnumbered.length === 0
  )
    return null;
  return (
    <div className="min-w-0 space-y-2">
      {written.length > 0 && (
        <aside
          className="rounded border border-rule bg-warn-soft p-3 text-sm"
          aria-label="Input not verified"
        >
          <p className="font-medium">Input not verified by Minerva</p>
          <p className="mt-1">
            This was computed from an equation the model wrote and Minerva never verified. The
            computation was checked, but that does not establish the input.
          </p>
          <ul className="mt-1 list-disc pl-5 text-xs">
            {written.map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ul>
        </aside>
      )}
      {parts.length > 0 && (
        <aside
          className="rounded border border-rule bg-warn-soft p-3 text-sm"
          aria-label="Input taken from a list"
        >
          <p className="font-medium">Input taken from a list</p>
          <p className="mt-1">
            The formula is exactly what the earlier tool returned, but that tool&apos;s verification
            and restrictions were not carried forward by reusing one component by number.
          </p>
          <ul className="mt-1 list-disc pl-5 text-xs">
            {parts.map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ul>
        </aside>
      )}
      {unnumbered.length > 0 && (
        <aside
          className="rounded border border-rule bg-warn-soft p-3 text-sm"
          aria-label="Input without a result id"
        >
          <p className="font-medium">Input without a result id</p>
          <p className="mt-1">
            A tool computed this equation and its formula is exact, but reusing it by number does
            not carry that tool&apos;s verification or restrictions.
          </p>
          <ul className="mt-1 list-disc pl-5 text-xs">
            {unnumbered.map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ul>
        </aside>
      )}
      {restrictions.length > 0 && (
        <aside
          className="min-w-0 rounded border border-rule bg-warn-soft p-3 text-sm"
          aria-label="Inherited conditions"
        >
          <p className="font-medium">Valid only under these conditions from earlier results</p>
          {restrictions.map((latex, i) => (
            <Math key={i} latex={latex} display />
          ))}
        </aside>
      )}
    </div>
  );
}
