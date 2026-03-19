"use client";

import { memo, useCallback } from "react";
import { Handle, Position, type NodeProps, type Node } from "@xyflow/react";
import { ChevronRight, ChevronDown } from "lucide-react";
import { cn, formatParams } from "@/lib/utils";
import { useStore } from "@/lib/store";
import type { ArchBlock } from "@/lib/ir";

export interface BlockNodeData {
  block: ArchBlock;
  isExpanded: boolean;
  isDiffed: boolean;
  diffDelta?: number;
}

const BLOCK_COLORS: Record<string, string> = {
  embedding: "border-blue-500/60 bg-blue-500/10",
  transformer_stack: "border-violet-500/60 bg-violet-500/10",
  multi_head_attention: "border-purple-400/60 bg-purple-400/10",
  feed_forward: "border-orange-400/60 bg-orange-400/10",
  moe_feed_forward: "border-amber-400/60 bg-amber-400/10",
  layer_norm: "border-slate-400/40 bg-slate-400/5",
  linear: "border-emerald-400/60 bg-emerald-400/10",
  ssm: "border-cyan-400/60 bg-cyan-400/10",
  conv1d: "border-teal-400/60 bg-teal-400/10",
  add: "border-green-500/50 bg-green-500/8",
  unknown: "border-slate-500/40 bg-slate-500/5",
};

const BLOCK_DOT: Record<string, string> = {
  embedding: "bg-blue-400",
  transformer_stack: "bg-violet-400",
  multi_head_attention: "bg-purple-400",
  feed_forward: "bg-orange-400",
  moe_feed_forward: "bg-amber-400",
  layer_norm: "bg-slate-400",
  linear: "bg-emerald-400",
  ssm: "bg-cyan-400",
  conv1d: "bg-teal-400",
  add: "bg-green-400",
  unknown: "bg-slate-400",
};

type BlockNodeType = Node<{ data: BlockNodeData }>;

function BlockNode({ data, selected }: NodeProps<BlockNodeType>) {
  const { block, isExpanded, isDiffed, diffDelta } = (data as { data: BlockNodeData }).data;
  const { toggleBlockExpanded, setSelectedBlockId } = useStore();

  const hasChildren = block.children.length > 0;
  const isStack = block.type === "transformer_stack";
  const colorClass =
    BLOCK_COLORS[block.type] ?? BLOCK_COLORS.unknown;
  const dotClass = BLOCK_DOT[block.type] ?? BLOCK_DOT.unknown;

  const handleExpand = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      if (hasChildren) toggleBlockExpanded(block.id);
    },
    [block.id, hasChildren, toggleBlockExpanded]
  );

  const handleSelect = useCallback(() => {
    setSelectedBlockId(block.id);
  }, [block.id, setSelectedBlockId]);

  // For container blocks (children present), param_count already covers all repeats.
  // For leaf blocks, param_count is per-instance and must be scaled by repeat.
  // Zero-param blocks (Add/Residual nodes) show no param line.
  const paramCount =
    block.param_count != null && block.param_count > 0
      ? formatParams(
          block.children.length > 0
            ? block.param_count
            : block.param_count * block.repeat
        )
      : null;

  return (
    <div
      onClick={handleSelect}
      className={cn(
        "relative rounded-lg border px-3 py-2 cursor-pointer transition-all min-w-[220px]",
        colorClass,
        selected && "ring-2 ring-white/50",
        isDiffed && "ring-2 ring-yellow-400/80"
      )}
    >
      <Handle type="target" position={Position.Top} className="!bg-slate-400 !border-slate-600" />
      {/* Hidden handles used exclusively for residual skip edges */}
      <Handle type="source" position={Position.Right} id="source-right" style={{ opacity: 0, pointerEvents: "none" }} />
      <Handle type="source" position={Position.Left}  id="source-left"  style={{ opacity: 0, pointerEvents: "none" }} />
      <Handle type="source" position={Position.Bottom} id="source-bottom" style={{ opacity: 0, pointerEvents: "none" }} />
      <Handle type="target" position={Position.Left}  id="target-left"  style={{ opacity: 0, pointerEvents: "none" }} />

      <div className="flex items-center gap-2">
        <div className={cn("w-2 h-2 rounded-full flex-shrink-0", dotClass)} />

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1">
            <span className="text-sm font-medium text-white">
              {block.label}
            </span>
            {isStack && block.repeat > 1 && (
              <span className="text-xs text-slate-400 flex-shrink-0">
                ×{block.repeat}
              </span>
            )}
          </div>
          {paramCount && (
            <span className="text-xs text-slate-400">{paramCount} params</span>
          )}
        </div>

        {hasChildren && (
          <button
            onClick={handleExpand}
            className="flex-shrink-0 text-slate-400 hover:text-white transition-colors"
          >
            {isExpanded ? (
              <ChevronDown className="w-3.5 h-3.5" />
            ) : (
              <ChevronRight className="w-3.5 h-3.5" />
            )}
          </button>
        )}
      </div>

      {isDiffed && diffDelta !== undefined && (
        <div
          className={cn(
            "absolute -top-2 -right-2 text-[10px] font-bold px-1.5 py-0.5 rounded-full",
            diffDelta > 0
              ? "bg-green-500 text-white"
              : "bg-red-500 text-white"
          )}
        >
          {diffDelta > 0 ? "+" : ""}
          {formatParams(diffDelta)}
        </div>
      )}

      <Handle type="source" position={Position.Bottom} className="!bg-slate-400 !border-slate-600" />
    </div>
  );
}

export default memo(BlockNode);
