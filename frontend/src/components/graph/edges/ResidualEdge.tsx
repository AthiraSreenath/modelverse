"use client";

import { memo } from "react";
import type { EdgeProps } from "@xyflow/react";

/**
 * Draws residual (skip) connections as explicit L-shaped or U-shaped paths.
 *
 *  Case A – source LEFT of target (encoder block → first Add):
 *    L-shape: │ down to Add(1) y, then → right.
 *    Vertical lives in the 40 px gap at sourceX (= encoder right edge, x≈240).
 *    Represents: Add1 = x + Attn(x)
 *
 *  Case B – source same x as target (LayerNorm → second Add):
 *    Tight U-shape: ← 16 px left, │ down to Add(2) y, → right.
 *    Vertical lives at targetX−16 (x≈264), clearly offset from Case A.
 *    Represents: Add2 = y + FFN(y)  where y = LayerNorm output
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
}: EdgeProps) {
  const STROKE = "#4ade80";
  const DASH = "5 3";

  let path: string;
  let labelX: number;
  let labelY: number;

  if (sourceX < targetX - 5) {
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
