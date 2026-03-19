"use client";

import { memo } from "react";
import type { EdgeProps } from "@xyflow/react";

/**
 * Draws residual (skip) connections as explicit L-shaped or U-shaped paths.
 *
 *  Case A – source LEFT of target (parent trunk → first Add, Post-LN style):
 *    L-shape: │ down to Add(1) y, then → right.
 *    Represents: Add1 = x + Attn(x)  (BERT) or Add1 = x + Attn(LN(x))  (Pre-LN from trunk)
 *
 *  Case B – source same x as target (e.g. LayerNorm → second Add in Post-LN):
 *    Tight U-shape: ← 16 px left, │ down to Add(2) y, → right.
 *    Represents: Add2 = y + FFN(y)  where y = LayerNorm output (BERT)
 *
 *  Case C – source-bottom (prior Add → next Add, Pre-LN / T5):
 *    │ down from bottom of upper Add, → into lower Add.
 *    Represents: Add2 = y + FFN(LN(y))  where y = output of first Add (not raw trunk x)
 *
 * A filled dot is drawn at the source point for both cases so the eye
 * immediately sees where each residual taps off from.
 */
function ResidualEdge({
  sourceX,
  sourceY,
  targetX,
  targetY,
  label,
  data,
}: EdgeProps) {
  const STROKE = "#4ade80";
  const DASH = "5 3";
  const fromPriorAddBottom = Boolean(
    data &&
      typeof data === "object" &&
      (data as { residualFromPriorAddBottom?: boolean }).residualFromPriorAddBottom
  );

  let path: string;
  let labelX: number;
  let labelY: number;

  if (fromPriorAddBottom) {
    // Pre-LN / T5: residual merges from the *output* of the previous Add (bottom handle).
    path = `M ${sourceX},${sourceY} V ${targetY} H ${targetX}`;
    labelX = sourceX + (targetX - sourceX) / 2;
    labelY = targetY - 10;
  } else if (sourceX < targetX - 5) {
    // ── Case A: encoder block (x) → first Add ─────────────────────────────
    // L-shape: ↓ down to Add(1) level, → right into Add(1).
    // Source dot sits at the encoder's right edge - the layer-entry tap point.
    path = `M ${sourceX},${sourceY} V ${targetY} H ${targetX}`;
    labelX = sourceX + (targetX - sourceX) / 2;
    labelY = targetY - 10;
  } else {
    // ── Case B: LayerNorm (y) → second Add ────────────────────────────────
    // Tight U-shape: ← 16 px, ↓ down to Add(2) level, → right into Add(2).
    // Vertical at targetX−16 is clearly distinct from Case A (at targetX−40).
    // Source dot sits at LayerNorm's left edge - the FFN-entry tap point.
    const LANE_OFFSET = 16;
    const laneX = targetX - LANE_OFFSET;
    path = `M ${sourceX},${sourceY} H ${laneX} V ${targetY} H ${targetX}`;
    labelX = laneX + LANE_OFFSET / 2;
    labelY = targetY - 10;
  }

  return (
    <g>
      {/* Wider invisible hit area */}
      <path d={path} stroke="transparent" strokeWidth={8} fill="none" />
      {/* Dashed green path */}
      <path
        d={path}
        stroke={STROKE}
        strokeWidth={1.5}
        strokeDasharray={DASH}
        fill="none"
      />
      {/* Filled dot at the source - shows exactly which node the residual taps */}
      <circle cx={sourceX} cy={sourceY} r={3} fill={STROKE} opacity={0.85} />
      {/* Label */}
      {label && (
        <text
          x={labelX}
          y={labelY}
          fill={STROKE}
          fontSize={9}
          textAnchor="middle"
          dominantBaseline="auto"
          style={{ pointerEvents: "none", userSelect: "none" }}
        >
          {label as string}
        </text>
      )}
    </g>
  );
}

export default memo(ResidualEdge);
