/** A small dependency-free SVG line chart: several series, axes with round ticks, a legend. */

export interface PlotSeries {
  label: string;
  x: number[];
  y: number[];
}

const W = 520;
const H = 300;
const M = { l: 58, r: 12, t: 12, b: 44 };
const DASHES = ["", "7 3", "2 3", "9 3 2 3", "1 4", "11 4"];

export function seriesColor(i: number): string {
  return `var(--series-${(i % 6) + 1})`;
}

/** Round tick positions (1, 2 or 5 times a power of ten) covering [min, max]. */
export function niceTicks(min: number, max: number, target = 5): number[] {
  if (!(max > min)) return [min];
  const raw = (max - min) / target;
  const power = Math.pow(10, Math.floor(Math.log10(raw)));
  const fraction = raw / power;
  const step = (fraction < 1.5 ? 1 : fraction < 3 ? 2 : fraction < 7 ? 5 : 10) * power;
  const first = Math.ceil(min / step - 1e-9);
  const ticks: number[] = [];
  for (let k = first; k * step <= max + step * 1e-9; k++) {
    ticks.push(Number((k * step).toPrecision(12)));
  }
  return ticks;
}

export function formatTick(v: number): string {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e4 || a < 1e-3) return v.toExponential(1).replace("e+", "e");
  return String(Number(v.toPrecision(4)));
}

function extent(values: number[]): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) {
    if (!Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (lo === Infinity) return [0, 1];
  if (lo === hi) {
    const pad = Math.abs(lo) * 0.1 || 1;
    return [lo - pad, hi + pad];
  }
  const pad = (hi - lo) * 0.04;
  return [lo - pad, hi + pad];
}

export function LinePlot({
  series,
  xLabel,
  yLabel,
  label,
  yFloor,
}: {
  series: PlotSeries[];
  xLabel: string;
  yLabel: string;
  /** Accessible name of the chart. */
  label: string;
  /** Start the y axis here (for quantities that cannot be negative, such as a drift). */
  yFloor?: number;
}) {
  const [x0, x1] = extent(series.flatMap((s) => s.x));
  const [lo, hi] = extent(series.flatMap((s) => s.y));
  const y0 = yFloor ?? lo;
  const y1 = yFloor === undefined ? hi : hi > yFloor ? hi : yFloor + 1;
  const px = (x: number) => M.l + ((x - x0) / (x1 - x0)) * (W - M.l - M.r);
  const py = (y: number) => H - M.b - ((y - y0) / (y1 - y0)) * (H - M.t - M.b);
  const xt = niceTicks(x0, x1);
  const yt = niceTicks(y0, y1);

  return (
    <figure className="mt-2">
      <div className="overflow-x-auto">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          role="img"
          aria-label={label}
          className="w-full text-muted"
          style={{ minWidth: 380, maxWidth: 640 }}
        >
          <title>{label}</title>
          {yt.map((v) => (
            <g key={`y${v}`}>
              <line x1={M.l} x2={W - M.r} y1={py(v)} y2={py(v)} stroke="var(--rule)" />
              <text x={M.l - 6} y={py(v) + 4.5} textAnchor="end" fontSize="13" fill="currentColor">
                {formatTick(v)}
              </text>
            </g>
          ))}
          {xt.map((v) => (
            <g key={`x${v}`}>
              <line x1={px(v)} x2={px(v)} y1={H - M.b} y2={H - M.b + 4} stroke="currentColor" />
              <text
                x={px(v)}
                y={H - M.b + 18}
                textAnchor="middle"
                fontSize="13"
                fill="currentColor"
              >
                {formatTick(v)}
              </text>
            </g>
          ))}
          <line x1={M.l} x2={W - M.r} y1={H - M.b} y2={H - M.b} stroke="currentColor" />
          <line x1={M.l} x2={M.l} y1={M.t} y2={H - M.b} stroke="currentColor" />
          <text
            x={(M.l + W - M.r) / 2}
            y={H - 5}
            textAnchor="middle"
            fontSize="13"
            fill="currentColor"
          >
            {xLabel}
          </text>
          <text
            x={12}
            y={(M.t + H - M.b) / 2}
            textAnchor="middle"
            fontSize="13"
            fill="currentColor"
            transform={`rotate(-90 12 ${(M.t + H - M.b) / 2})`}
          >
            {yLabel}
          </text>
          {series.map((s, i) => {
            const d = s.x
              .map((x, k) => ({ x, y: s.y[k] }))
              .filter(
                (p): p is { x: number; y: number } => p.y !== undefined && Number.isFinite(p.y),
              )
              .map((p, k) => `${k === 0 ? "M" : "L"}${px(p.x).toFixed(1)} ${py(p.y).toFixed(1)}`)
              .join(" ");
            return (
              <path
                key={s.label}
                d={d}
                fill="none"
                stroke={seriesColor(i)}
                strokeWidth={1.8}
                strokeDasharray={DASHES[i % DASHES.length]}
                strokeLinejoin="round"
              />
            );
          })}
        </svg>
      </div>
      <figcaption className="flex flex-wrap gap-x-4 gap-y-1 px-1 text-xs text-muted">
        {series.map((s, i) => (
          <span key={s.label} className="inline-flex items-center gap-1.5">
            <svg width="26" height="8" aria-hidden="true">
              <line
                x1="0"
                x2="26"
                y1="4"
                y2="4"
                stroke={seriesColor(i)}
                strokeWidth={2}
                strokeDasharray={DASHES[i % DASHES.length]}
              />
            </svg>
            {s.label}
          </span>
        ))}
      </figcaption>
    </figure>
  );
}
