import type { PlotSeries } from "./line-plot";
import { formatTick, niceTicks, seriesColor } from "./line-plot";

/** A direction field with equilibria and trajectories, as a dependency-free SVG. */

const W = 520;
const H = 380;
const M = { l: 58, r: 14, t: 12, b: 46 };

export interface FieldPoints {
  window: number[];
  x: number[];
  y: number[];
  dx: number[];
  dy: number[];
}

export interface Marker {
  x: number;
  y: number;
  label: string;
  tone: "ok" | "warn" | "muted";
}

const TONE = { ok: "var(--ok)", warn: "var(--warn)", muted: "var(--ink-muted)" } as const;

export function PhasePlot({
  field,
  markers = [],
  trajectories = [],
  xLabel,
  yLabel,
  label,
}: {
  field: FieldPoints;
  markers?: Marker[];
  trajectories?: PlotSeries[];
  xLabel: string;
  yLabel: string;
  label: string;
}) {
  const [x0, x1, y0, y1] = field.window as [number, number, number, number];
  const px = (x: number) => M.l + ((x - x0) / (x1 - x0)) * (W - M.l - M.r);
  const py = (y: number) => H - M.b - ((y - y0) / (y1 - y0)) * (H - M.t - M.b);
  const columns = new Set(field.x).size || 1;
  const cell = (W - M.l - M.r) / columns;
  const clip = `phase-clip-${label.replace(/\W+/g, "-")}`;
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
          <defs>
            <clipPath id={clip}>
              <rect x={M.l} y={M.t} width={W - M.l - M.r} height={H - M.t - M.b} />
            </clipPath>
          </defs>
          {niceTicks(y0, y1).map((v) => (
            <g key={`y${v}`}>
              <line x1={M.l} x2={W - M.r} y1={py(v)} y2={py(v)} stroke="var(--rule)" />
              <text x={M.l - 6} y={py(v) + 4.5} textAnchor="end" fontSize="13" fill="currentColor">
                {formatTick(v)}
              </text>
            </g>
          ))}
          {niceTicks(x0, x1).map((v) => (
            <g key={`x${v}`}>
              <line x1={px(v)} x2={px(v)} y1={M.t} y2={H - M.b} stroke="var(--rule)" />
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
            y={H - 6}
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
          <g clipPath={`url(#${clip})`}>
            {field.x.map((x, i) => {
              const dx = field.dx[i] ?? 0;
              const dy = field.dy[i] ?? 0;
              // The arrow's direction on the screen, with the axes' different scales taken into account.
              const sx = dx / (x1 - x0);
              const sy = dy / (y1 - y0);
              const length = Math.hypot(sx * (W - M.l - M.r), sy * (H - M.t - M.b));
              if (!(length > 1e-9)) {
                return (
                  <circle key={i} cx={px(x)} cy={py(field.y[i] ?? 0)} r={1.2} fill="currentColor" />
                );
              }
              const ux = (sx * (W - M.l - M.r)) / length;
              const uy = -(sy * (H - M.t - M.b)) / length;
              const half = cell * 0.36;
              const cx = px(x);
              const cy = py(field.y[i] ?? 0);
              const tipX = cx + ux * half;
              const tipY = cy + uy * half;
              const back = 4.5;
              return (
                <g key={i} stroke="currentColor" strokeWidth={1} opacity={0.75}>
                  <line x1={cx - ux * half} y1={cy - uy * half} x2={tipX} y2={tipY} />
                  <line
                    x1={tipX}
                    y1={tipY}
                    x2={tipX - back * (ux * 0.87 - uy * 0.5)}
                    y2={tipY - back * (uy * 0.87 + ux * 0.5)}
                  />
                  <line
                    x1={tipX}
                    y1={tipY}
                    x2={tipX - back * (ux * 0.87 + uy * 0.5)}
                    y2={tipY - back * (uy * 0.87 - ux * 0.5)}
                  />
                </g>
              );
            })}
            {trajectories.map((s, i) => {
              const d = s.x
                .map((x, k) => ({ x, y: s.y[k] }))
                .filter(
                  (p): p is { x: number; y: number } => p.y !== undefined && Number.isFinite(p.y),
                )
                .map((p, k) => `${k === 0 ? "M" : "L"}${px(p.x).toFixed(1)} ${py(p.y).toFixed(1)}`)
                .join(" ");
              const first = { x: s.x[0] ?? 0, y: s.y[0] ?? 0 };
              return (
                <g key={s.label} data-trajectory={s.label}>
                  <path d={d} fill="none" stroke={seriesColor(i)} strokeWidth={2} />
                  <circle cx={px(first.x)} cy={py(first.y)} r={3.5} fill={seriesColor(i)} />
                </g>
              );
            })}
            {markers.map((m) => (
              <g key={m.label}>
                <circle
                  cx={px(m.x)}
                  cy={py(m.y)}
                  r={5.5}
                  fill="var(--paper)"
                  stroke={TONE[m.tone]}
                  strokeWidth={2.5}
                />
              </g>
            ))}
          </g>
        </svg>
      </div>
      <figcaption className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted">
        <span>arrows: the direction of the field (a picture of f, not of a solution)</span>
        {markers.map((m) => (
          <span key={m.label} style={{ color: TONE[m.tone] }}>
            ○ {m.label}
          </span>
        ))}
        {trajectories.map((s, i) => (
          <span key={s.label} style={{ color: seriesColor(i) }}>
            ● {s.label}
          </span>
        ))}
      </figcaption>
    </figure>
  );
}
