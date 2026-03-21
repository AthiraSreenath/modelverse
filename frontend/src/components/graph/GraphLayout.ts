/**
 * Converts ArchitectureIR blocks into React Flow nodes + edges.
 * Uses a simple top-down layout (no Dagre needed for linear architectures).
 *
 * Layout hints on ArchBlock drive non-linear graphs:
 *   same_row_as  — place this block alongside (same Y, offset X) a named block.
 *                  Used for parallel branches (e.g. CLIP ‖ SAM in VLMs).
 *   merge_from   — explicit incoming edge sources; overrides auto-connect.
 *                  Empty array = branch start (no incoming edges).
 */

import type { Node, Edge } from "@xyflow/react";
import type { ArchBlock, ArchitectureIR, DiffEntry } from "@/lib/ir";
import type { BlockNodeData } from "./nodes/BlockNode";
import type { RepeatLabelData } from "./nodes/RepeatLabelNode";

const NODE_WIDTH    = 240;
const NODE_HEIGHT   = 64;
const VERT_GAP      = 24;
const PARALLEL_GAP  = 20;   // horizontal gap between side-by-side parallel blocks
const CHILD_X_OFFSET  = 280;
const CHILD_VERT_GAP  = 12;

/** Get block IDs that appear in the diff */
function getDiffedBlockIds(diff: DiffEntry[]): Map<string, number> {
  const map = new Map<string, number>();
  for (const entry of diff) {
    const parts = entry.path.split(".");
    const blockId = parts[0];
    if (blockId) {
      const oldVal = typeof entry.old === "number" ? entry.old : null;
      const newVal = typeof entry.new === "number" ? entry.new : null;
      if (oldVal !== null && newVal !== null) {
        map.set(blockId, newVal - oldVal);
      } else {
        map.set(blockId, 0);
      }
    }
  }
  return map;
}

export function buildGraphElements(
  ir: ArchitectureIR,
  expandedBlockIds: Set<string>,
  diff: DiffEntry[]
): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = [];
  const edges: Edge[] = [];
  const diffedIds = getDiffedBlockIds(diff);

  let y = 0;
  let prevId: string | null = null;

  // Track placed position of each top-level block for layout hints
  const yById = new Map<string, number>();
  const xById = new Map<string, number>();

  function addBlock(
    block: ArchBlock,
    x: number,
    parentId?: string
  ): number {
    const isExpanded = expandedBlockIds.has(block.id);
    const isDiffed = diffedIds.has(block.id);
    const diffDelta = diffedIds.get(block.id);

    // ── Determine position ──────────────────────────────────────────────────
    let blockX = x;
    let blockY = y;

    if (!parentId && block.same_row_as) {
      // Parallel sibling: place at same Y as the referenced block, offset right
      const refY = yById.get(block.same_row_as) ?? y;
      const refX = xById.get(block.same_row_as) ?? 0;
      // Stack siblings further right if the referenced block itself was a sibling
      blockX = refX + NODE_WIDTH + PARALLEL_GAP;
      blockY = refY;
    } else if (!parentId && block.merge_from != null) {
      // Convergence point: sit below the deepest of its sources
      let maxSourceBottom = 0;
      for (const srcId of block.merge_from) {
        const srcY = yById.get(srcId) ?? 0;
        maxSourceBottom = Math.max(maxSourceBottom, srcY + NODE_HEIGHT);
      }
      blockY = Math.max(y, maxSourceBottom + VERT_GAP);
      blockX = 0;
    }

    const nodeData: BlockNodeData = {
      block,
      isExpanded,
      isDiffed,
      diffDelta,
    };

    nodes.push({
      id: block.id,
      type: "blockNode",
      position: { x: blockX, y: blockY },
      data: { data: nodeData },
      style: { width: NODE_WIDTH },
    });

    const nodeBottomY = blockY + NODE_HEIGHT;

    // ── Edges ───────────────────────────────────────────────────────────────
    if (!parentId) {
      if (block.merge_from != null) {
        // Explicit multi-source edges
        for (const srcId of block.merge_from) {
          edges.push({
            id: `${srcId}->${block.id}`,
            source: srcId,
            target: block.id,
            type: "smoothstep",
            style: { stroke: "#475569", strokeWidth: 1.5 },
          });
        }
      } else if (!block.same_row_as && prevId) {
        // Normal linear auto-connect (skip for parallel siblings)
        edges.push({
          id: `${prevId}->${block.id}`,
          source: prevId,
          target: block.id,
          type: "smoothstep",
          style: { stroke: "#475569", strokeWidth: 1.5 },
        });
      }
    }

    // ── Update tracking ─────────────────────────────────────────────────────
    if (!parentId) {
      yById.set(block.id, blockY);
      xById.set(block.id, blockX);

      if (block.same_row_as) {
        // Parallel sibling: don't advance y or update prevId
      } else {
        // Main chain: advance y and update prevId
        y = blockY + NODE_HEIGHT + VERT_GAP;
        prevId = block.id;
      }
    } else {
      // Child block inside expanded stack: y is managed by the parent loop
      prevId = block.id;
    }

    const entryY = blockY;

    // ── Expanded children ───────────────────────────────────────────────────
    if (isExpanded && block.children.length > 0) {
      const childX = blockX + CHILD_X_OFFSET;
      let childPrevId: string | null = null;
      const childrenStartY = y;

      const rawLayout = block.params?.residual_layout as string | undefined;
      const residualLayout =
        rawLayout ||
        (block.children[0]?.type === "layer_norm" ? "pre_ln" : "post_ln");
      const anchorAfterLayerNorm = residualLayout === "post_ln";
      let residualAnchor: string = block.id;
      let pendingSkipSrc: string = block.id;

      for (const child of block.children) {
        const childIsExpanded = expandedBlockIds.has(child.id);
        const childNodeData: BlockNodeData = {
          block: child,
          isExpanded: childIsExpanded,
          isDiffed: diffedIds.has(child.id),
          diffDelta: diffedIds.get(child.id),
        };
        const childNodeId = `${block.id}__${child.id}`;

        nodes.push({
          id: childNodeId,
          type: "blockNode",
          position: { x: childX, y },
          data: { data: childNodeData },
          style: { width: NODE_WIDTH },
        });

        if (childPrevId) {
          edges.push({
            id: `${childPrevId}->${childNodeId}`,
            source: childPrevId,
            target: childNodeId,
            type: "smoothstep",
            style: { stroke: "#6366f1", strokeWidth: 1 },
          });
        } else {
          edges.push({
            id: `${block.id}->${childNodeId}`,
            source: block.id,
            target: childNodeId,
            type: "smoothstep",
            style: { stroke: "#6366f1", strokeWidth: 1, strokeDasharray: "4" },
          });
        }

        const ct = child.type;
        if (
          ct === "multi_head_attention" ||
          ct === "feed_forward" ||
          ct === "moe_feed_forward"
        ) {
          pendingSkipSrc = residualAnchor;
        } else if (ct === "add") {
          const isParentSrc = pendingSkipSrc === block.id;
          const isFromPriorAdd =
            !isParentSrc && pendingSkipSrc.includes("__add_");
          const useBottomSource =
            isFromPriorAdd &&
            (residualLayout === "pre_ln" || residualLayout === "t5_decoder");
          const skipLabel = isParentSrc ? "x" : "y";
          edges.push({
            id: `skip::${pendingSkipSrc}::${childNodeId}`,
            source: pendingSkipSrc,
            target: childNodeId,
            sourceHandle: isParentSrc
              ? "source-right"
              : useBottomSource
                ? "source-bottom"
                : "source-left",
            targetHandle: "target-left",
            type: "residual",
            label: skipLabel,
            data: { residualFromPriorAddBottom: useBottomSource },
          });
          residualAnchor = childNodeId;
        } else if (ct === "layer_norm" && anchorAfterLayerNorm) {
          residualAnchor = childNodeId;
        }

        childPrevId = childNodeId;
        y += NODE_HEIGHT + CHILD_VERT_GAP;

        if (childIsExpanded && child.children.length > 0) {
          const gcX = childX + CHILD_X_OFFSET;
          let gcPrevId: string | null = null;
          const gcParentNodeId = `${block.id}__${child.id}`;

          for (const gc of child.children) {
            const gcId = `${block.id}__${child.id}__${gc.id}`;
            const gcData: BlockNodeData = {
              block: gc,
              isExpanded: false,
              isDiffed: diffedIds.has(gc.id),
              diffDelta: diffedIds.get(gc.id),
            };

            nodes.push({
              id: gcId,
              type: "blockNode",
              position: { x: gcX, y },
              data: { data: gcData },
              style: { width: NODE_WIDTH },
            });

            if (gcPrevId) {
              edges.push({
                id: `${gcPrevId}->${gcId}`,
                source: gcPrevId,
                target: gcId,
                type: "smoothstep",
                style: { stroke: "#a855f7", strokeWidth: 1 },
              });
            } else {
              edges.push({
                id: `${gcParentNodeId}->${gcId}`,
                source: gcParentNodeId,
                target: gcId,
                type: "smoothstep",
                style: { stroke: "#a855f7", strokeWidth: 1, strokeDasharray: "4" },
              });
            }

            gcPrevId = gcId;
            y += NODE_HEIGHT + CHILD_VERT_GAP;
          }
          y += CHILD_VERT_GAP;
        }
      }

      if (block.repeat > 1) {
        const BRACKET_PAD = 8;
        const childrenActualHeight = y - CHILD_VERT_GAP - childrenStartY;
        const labelData: RepeatLabelData = {
          repeat: block.repeat,
          heightPx: childrenActualHeight + BRACKET_PAD * 2,
        };
        nodes.push({
          id: `${block.id}__repeat_label`,
          type: "repeatLabel",
          position: { x: childX + NODE_WIDTH + 16, y: childrenStartY - BRACKET_PAD },
          data: labelData,
          draggable: false,
          selectable: false,
          focusable: false,
        });
      }

      y += VERT_GAP;
    }

    return entryY;
  }

  for (const block of ir.blocks) {
    addBlock(block, 0);
  }

  return { nodes, edges };
}
