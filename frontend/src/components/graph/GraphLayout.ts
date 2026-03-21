/**
 * Converts ArchitectureIR blocks into React Flow nodes + edges.
 *
 * Default: top-down single-column layout for standard transformer models.
 *
 * Multi-column layout (VLMs etc.) driven by ArchBlock layout hints:
 *   layout_column  — which horizontal column a block lives in.
 *                    col 0 = left/main, col 1 = right/vision, etc.
 *                    Each column has its own independent y cursor.
 *   same_row_as    — place this block at the same Y as the named block,
 *                    offset to the right. Used for CLIP ‖ SAM etc.
 *   merge_from     — explicit incoming edge list. Overrides auto-connect.
 *                    [] = branch start (no incoming edge).
 *
 * Result for a 2-col VLM:
 *
 *   col 0 (x=0)          col 1 (x=280)
 *   [Token Embeddings]   [CLIP ViT-L/14]  [SAM ViT-B]
 *          │              ╲              ╱
 *          │               [Feat Fusion]
 *          │               [Vis Projector]
 *          │              ╱
 *   [LM Decoder] ◄────────
 *   [Final Norm]
 *   [LM Head]
 *
 * Every edge from col 0 to LM Decoder runs down the LEFT column,
 * cleanly separated from the vision branch on the right.
 */

import type { Node, Edge } from "@xyflow/react";
import type { ArchBlock, ArchitectureIR, DiffEntry } from "@/lib/ir";
import type { BlockNodeData } from "./nodes/BlockNode";
import type { RepeatLabelData } from "./nodes/RepeatLabelNode";

const NODE_WIDTH    = 240;
const NODE_HEIGHT   = 80;    // generous — nodes are variable height; this is min spacing
const VERT_GAP      = 32;    // vertical gap between consecutive blocks in a column
const COL_MARGIN    = 40;    // horizontal gap between columns
const PARALLEL_GAP  = 20;    // gap between same_row_as siblings within a column
const CHILD_X_OFFSET  = 280;
const CHILD_VERT_GAP  = 12;

function getDiffedBlockIds(diff: DiffEntry[]): Map<string, number> {
  const map = new Map<string, number>();
  for (const entry of diff) {
    const parts = entry.path.split(".");
    const blockId = parts[0];
    if (blockId) {
      const oldVal = typeof entry.old === "number" ? entry.old : null;
      const newVal = typeof entry.new === "number" ? entry.new : null;
      map.set(blockId, oldVal !== null && newVal !== null ? newVal - oldVal : 0);
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

  // Per-column state
  const yByCol    = new Map<number, number>();   // y cursor per column
  const prevByCol = new Map<number, string | null>(); // last block id per column
  // Per-block position lookup (for merge_from / same_row_as)
  const yById = new Map<string, number>();
  const xById = new Map<string, number>();

  const getColY    = (col: number) => yByCol.get(col) ?? 0;
  const getColPrev = (col: number) => prevByCol.get(col) ?? null;

  function addTopLevelBlock(block: ArchBlock): void {
    const col: number = typeof block.layout_column === "number" ? block.layout_column : 0;
    const colPrev = getColPrev(col);
    const colY    = getColY(col);

    // ── Determine position ──────────────────────────────────────────────────
    let blockX: number;
    let blockY: number;
    let isSidecar = false;  // same_row_as blocks don't advance their column

    if (block.same_row_as) {
      // Parallel sibling: same Y as the referenced block, shifted right
      const refY = yById.get(block.same_row_as) ?? colY;
      const refX = xById.get(block.same_row_as) ?? col * (NODE_WIDTH + COL_MARGIN);
      blockX = refX + NODE_WIDTH + PARALLEL_GAP;
      blockY = refY;
      isSidecar = true;

    } else if (block.merge_from != null) {
      // Convergence: place below the deepest source block
      let maxBottom = 0;
      for (const srcId of block.merge_from) {
        maxBottom = Math.max(maxBottom, (yById.get(srcId) ?? 0) + NODE_HEIGHT);
      }
      blockY = Math.max(colY, maxBottom + VERT_GAP);
      blockX = col * (NODE_WIDTH + COL_MARGIN);

    } else {
      // Normal column placement
      blockY = colY;
      blockX = col * (NODE_WIDTH + COL_MARGIN);
    }

    // ── Record position before placing ─────────────────────────────────────
    yById.set(block.id, blockY);
    xById.set(block.id, blockX);

    // ── Place node ──────────────────────────────────────────────────────────
    const isExpanded = expandedBlockIds.has(block.id);
    const isDiffed   = diffedIds.has(block.id);
    const diffDelta  = diffedIds.get(block.id);

    const nodeData: BlockNodeData = { block, isExpanded, isDiffed, diffDelta };
    nodes.push({
      id: block.id,
      type: "blockNode",
      position: { x: blockX, y: blockY },
      data: { data: nodeData },
      style: { width: NODE_WIDTH },
    });

    // ── Draw incoming edges ─────────────────────────────────────────────────
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
    } else if (!isSidecar && colPrev) {
      // Auto-connect from previous block in this column
      edges.push({
        id: `${colPrev}->${block.id}`,
        source: colPrev,
        target: block.id,
        type: "smoothstep",
        style: { stroke: "#475569", strokeWidth: 1.5 },
      });
    }

    // ── Advance column state ────────────────────────────────────────────────
    if (!isSidecar) {
      yByCol.set(col, blockY + NODE_HEIGHT + VERT_GAP);
      prevByCol.set(col, block.id);
    }

    // ── Expanded children (sub-blocks within a transformer stack) ───────────
    if (isExpanded && block.children.length > 0) {
      const childX = blockX + CHILD_X_OFFSET;
      let childPrevId: string | null = null;
      // Children start at the column's current y (just advanced above)
      const childrenStartY = getColY(col);

      // Residual skip-connection tracking
      const rawLayout = block.params?.residual_layout as string | undefined;
      const residualLayout =
        rawLayout ||
        (block.children[0]?.type === "layer_norm" ? "pre_ln" : "post_ln");
      const anchorAfterLayerNorm = residualLayout === "post_ln";
      let residualAnchor: string = block.id;
      let pendingSkipSrc: string = block.id;

      // local child y cursor
      let cy = childrenStartY;

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
          position: { x: childX, y: cy },
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

        // Residual skip edges
        const ct = child.type;
        if (ct === "multi_head_attention" || ct === "feed_forward" || ct === "moe_feed_forward") {
          pendingSkipSrc = residualAnchor;
        } else if (ct === "add") {
          const isParentSrc  = pendingSkipSrc === block.id;
          const isFromPriorAdd = !isParentSrc && pendingSkipSrc.includes("__add_");
          const useBottomSource =
            isFromPriorAdd &&
            (residualLayout === "pre_ln" || residualLayout === "t5_decoder");
          edges.push({
            id: `skip::${pendingSkipSrc}::${childNodeId}`,
            source: pendingSkipSrc,
            target: childNodeId,
            sourceHandle: isParentSrc ? "source-right" : useBottomSource ? "source-bottom" : "source-left",
            targetHandle: "target-left",
            type: "residual",
            label: isParentSrc ? "x" : "y",
            data: { residualFromPriorAddBottom: useBottomSource },
          });
          residualAnchor = childNodeId;
        } else if (ct === "layer_norm" && anchorAfterLayerNorm) {
          residualAnchor = childNodeId;
        }

        childPrevId = childNodeId;
        cy += NODE_HEIGHT + CHILD_VERT_GAP;

        // Grandchild expansion
        if (childIsExpanded && child.children.length > 0) {
          const gcX = childX + CHILD_X_OFFSET;
          let gcPrevId: string | null = null;
          const gcParentId = `${block.id}__${child.id}`;

          for (const gc of child.children) {
            const gcId = `${block.id}__${child.id}__${gc.id}`;
            nodes.push({
              id: gcId,
              type: "blockNode",
              position: { x: gcX, y: cy },
              data: { data: { block: gc, isExpanded: false, isDiffed: diffedIds.has(gc.id), diffDelta: diffedIds.get(gc.id) } },
              style: { width: NODE_WIDTH },
            });
            edges.push({
              id: gcPrevId ? `${gcPrevId}->${gcId}` : `${gcParentId}->${gcId}`,
              source: gcPrevId ?? gcParentId,
              target: gcId,
              type: "smoothstep",
              style: { stroke: "#a855f7", strokeWidth: 1, ...(gcPrevId ? {} : { strokeDasharray: "4" }) },
            });
            gcPrevId = gcId;
            cy += NODE_HEIGHT + CHILD_VERT_GAP;
          }
          cy += CHILD_VERT_GAP;
        }
      }

      // Repeat label
      if (block.repeat > 1) {
        const BRACKET_PAD = 8;
        const childrenActualHeight = cy - CHILD_VERT_GAP - childrenStartY;
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

      // Advance this column's y past the expanded children
      const childrenEndY = cy + VERT_GAP;
      yByCol.set(col, Math.max(getColY(col), childrenEndY));
    }
  }

  for (const block of ir.blocks) {
    addTopLevelBlock(block);
  }

  return { nodes, edges };
}
