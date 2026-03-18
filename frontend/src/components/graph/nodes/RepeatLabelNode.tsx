"use client";

import { memo } from "react";
import type { NodeProps, Node } from "@xyflow/react";

export interface RepeatLabelData extends Record<string, unknown> {
  repeat: number;
  heightPx: number;
}

type RepeatLabelNodeType = Node<RepeatLabelData>;

function RepeatLabelNode({ data }: NodeProps<RepeatLabelNodeType>) {
  const { repeat, heightPx } = data;

  return (
    <div
      className="flex items-center gap-2 pointer-events-none select-none"
      style={{ height: heightPx }}
    >
      {/* Label */}
      <div className="flex flex-col justify-center gap-0.5">
        <span className="text-2xl font-bold text-slate-400 leading-none">
          ×{repeat}
        </span>
        <span className="text-[10px] text-slate-600 leading-tight whitespace-nowrap">
          all layers identical
        </span>
      </div>

      {/* Bracket — vertical line on right side, serifs pointing left toward the nodes */}
      <div className="relative flex-shrink-0" style={{ height: heightPx, width: 10 }}>
        {/* Top serif */}
        <div className="absolute top-0 right-0 w-2.5 h-px bg-slate-600" />
        {/* Vertical line */}
        <div className="absolute top-0 right-0 w-px bg-slate-600" style={{ height: heightPx }} />
        {/* Bottom serif */}
        <div className="absolute bottom-0 right-0 w-2.5 h-px bg-slate-600" />
      </div>
    </div>
  );
}

export default memo(RepeatLabelNode);
