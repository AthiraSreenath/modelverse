/**
 * Converts ArchitectureIR blocks into React Flow nodes + edges.
 * Uses a simple top-down layout (no Dagre needed for linear architectures).
 */

import type { Node, Edge } from "@xyflow/react";
import type { ArchBlock, ArchitectureIR, DiffEntry } from "@/lib/ir";
import type { BlockNodeData } from "./nodes/BlockNode";
import type { RepeatLabelData } from "./nodes/RepeatLabelNode";

const NODE_WIDTH = 200;
const NODE_HEIGHT = 64;
const VERT_GAP = 24;
const CHILD_X_OFFSET = 240;
const CHILD_VERT_GAP = 12;

/** Get block IDs that appear in the diff */
function getDiffedBlockIds(diff: DiffEntry[]): Map<string, number> {
  const map = new Map<string, number>();
  for (const entry of diff) {
    const parts = entry.path.split(".");
    const blockId = parts[0];
    if (blockId) {
      // try to extract a numeric delta
      const oldVal =
        typeof entry.old === "number" ? entry.old : null;
      const newVal =
        typeof entry.new === "number" ? entry.new : null;
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

  function addBlock(
    block: ArchBlock,
    x: number,
    parentId?: string
  ): number {
    const isExpanded = expandedBlockIds.has(block.id);
    const isDiffed = diffedIds.has(block.id);
    const diffDelta = diffedIds.get(block.id);

    const nodeData: BlockNodeData = {
      block,
      isExpanded,
      isDiffed,
      diffDelta,
    };

    nodes.push({
      id: block.id,
      type: "blockNode",
      position: { x, y },
      data: { data: nodeData },
      style: { width: NODE_WIDTH },
    });

    const nodeBottomY = y + NODE_HEIGHT;

    if (prevId && !parentId) {
      edges.push({
        id: `${prevId}->${block.id}`,
        source: prevId,
        target: block.id,
        type: "smoothstep",
        style: { stroke: "#475569", strokeWidth: 1.5 },
      });
    }

    prevId = block.id;
    const entryY = y;
    y += NODE_HEIGHT + VERT_GAP;

    if (isExpanded && block.children.length > 0) {
      const childX = x + CHILD_X_OFFSET;
      let childPrevId: string | null = null;
      const childrenStartY = y;

      for (const child of block.children) {
        const childIsExpanded = expandedBlockIds.has(child.id);
        const childNodeData: BlockNodeData = {
          block: child,
          isExpanded: childIsExpanded,
          isDiffed: diffedIds.has(child.id),
          diffDelta: diffedIds.get(child.id),
        };

        nodes.push({
          id: `${block.id}__${child.id}`,
          type: "blockNode",
          position: { x: childX, y },
          data: { data: childNodeData },
          style: { width: NODE_WIDTH },
        });

        if (childPrevId) {
          edges.push({
            id: `${childPrevId}->${block.id}__${child.id}`,
            source: childPrevId,
            target: `${block.id}__${child.id}`,
            type: "smoothstep",
            style: { stroke: "#6366f1", strokeWidth: 1 },
          });
        } else {
          edges.push({
            id: `${block.id}->${block.id}__${child.id}`,
            source: block.id,
            target: `${block.id}__${child.id}`,
            type: "smoothstep",
            style: { stroke: "#6366f1", strokeWidth: 1, strokeDasharray: "4" },
          });
        }

        childPrevId = `${block.id}__${child.id}`;
        y += NODE_HEIGHT + CHILD_VERT_GAP;

        // ── Grandchild expansion (e.g. MoE FFN → Router / Expert FFNs / Combine) ──
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
          y += CHILD_VERT_GAP; // extra breathing room after grandchildren
        }
      }

      // Repeat label — only shown when the block repeats more than once.
      // Height is derived from the actual y cursor so it matches the rendered
      // positions exactly, regardless of NODE_HEIGHT accuracy.
      if (block.repeat > 1) {
        const BRACKET_PAD = 8; // px above first child and below last child
        // y is now at: childrenStartY + n*(NODE_HEIGHT+CHILD_VERT_GAP)
        // last child bottom = y - CHILD_VERT_GAP
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
