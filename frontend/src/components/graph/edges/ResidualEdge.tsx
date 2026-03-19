"use client";

import { memo } from "react";
import type { EdgeProps } from "@xyflow/react";

/**
 * Draws residual (skip) connections as explicit L-shaped or U-shaped paths
 * so the origin of each skip is unambiguous:
 *
 *  Case A – source is LEFT of target (encoder block → first Add):
 *    Vertical line down from source, then horizontal right into Add.
 *    Visually: a tall │ from the layer-entry level + a ├──→ branch.
 *
 *  Case B – source is same x as target (LayerNorm → second Add):
 *    Short horizontal left, vertical down, horizontal right into Add.
 *    Visually: a compact U-shape showing the FFN-entry tap point.
 *
 * Both paths stay inside the 40 px gap between the parent column and the
 * child column, never overlapping with any node.
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
  const LANE_OFFSET = 28; // how far left the U-curve goes

  let path: string;
  let labelX: number;
  let labelY: number;

  if (sourceX < targetX - 5) {
    // Case A — source is to the LEFT of the target column (parent block → Add)
    // L-shape: ↓ down to target Y, then → right into target
    path = `M ${sourceX},${sourceY} V ${targetY} H ${targetX}`;
    // "x" label on the short horizontal segment
    labelX = sourceX + (targetX - sourceX) / 2;
    labelY = targetY - 10;
  } else {
    // Case B — source and target share the same x (sibling → Add)
    // U-shape: ← left to lane, ↓ down to target Y, → right into target
    const laneX = targetX - LANE_OFFSET;
    path = `M ${sourceX},${sourceY} H ${laneX} V ${targetY} H ${targetX}`;
    // "x" label on the right horizontal segment
    labelX = laneX + LANE_OFFSET / 2;
    labelY = targetY - 10;
  }

  return (
    <g>
      {/* Wider invisible hit area */}
      <path d={path} stroke="transparent" strokeWidth={8} fill="none" />
      {/* Visible dashed green path */}
      <path
        d={path}
        stroke={STROKE}
        strokeWidth={1.5}
        strokeDasharray={DASH}
        fill="none"
      />
      {/* "x" label */}
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
