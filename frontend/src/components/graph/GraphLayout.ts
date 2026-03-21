/**
 * Converts ArchitectureIR blocks into React Flow nodes + edges.
 *
 * Positioning is delegated entirely to Dagre, the standard directed-graph
 * layout engine used with React Flow.  Dagre receives the block graph
 * (nodes + edges derived from merge_from / auto-connect) and returns correct
 * (x, y) coordinates that handle parallel branches, converging paths, and
 * variable-rank depths automatically — no manual pixel arithmetic required.
 *
 * The only hand-placed nodes are expanded children (sub-blocks within a
 * transformer stack), which are placed in a linear column to the right of
 * their parent's Dagre-assigned position.
 */

import dagre from "dagre";
import type { Node, Edge } from "@xyflow/react";
import type { ArchBlock, ArchitectureIR, DiffEntry } from "@/lib/ir";
import type { BlockNodeData } from "./nodes/BlockNode";
import type { RepeatLabelData } from "./nodes/RepeatLabelNode";

// Dagre node size. Width matches BlockNode's min-w-[220px] + padding.
// Height is a generous overestimate so Dagre reserves enough vertical space.
// (Actual rendered height can be less; extra gap is fine.)
const NODE_W       = 260;
const NODE_H       = 100;
const RANK_SEP     = 50;  // vertical gap between Dagre ranks
const NODE_SEP     = 40;  // horizontal gap between nodes in the same rank

// Expanded children layout (independent of Dagre)
const CHILD_X_OFFSET  = 300;
const CHILD_W         = 240;
const CHILD_H         = 80;
const CHILD_VERT_GAP  = 12;
const CHILD_RANK_SEP  = 16;

function getDiffedBlockIds(diff: DiffEntry[]): Map<string, number> {
  const map = new Map<string, number>();
  for (const entry of diff) {
    const blockId = entry.path.split(".")[0];
    if (blockId) {
      const oldVal = typeof entry.old === "number" ? entry.old : null;
      const newVal = typeof entry.new === "number" ? entry.new : null;
      map.set(blockId, oldVal !== null && newVal !== null ? newVal - oldVal : 0);
    }
  }
  return map;
}

/** Derive the edge list from block layout hints + sequential auto-connect. */
function deriveEdges(blocks: ArchBlock[]): Array<{ src: string; tgt: string }> {
  const result: Array<{ src: string; tgt: string }> = [];
  // Per-column previous-block tracking for auto-connect
  const colPrev = new Map<number, string>();

  for (const block of blocks) {
    const col = block.layout_column ?? 0;

    if (block.merge_from != null) {
      // Explicit incoming edges (possibly empty for branch starts)
      for (const srcId of block.merge_from) {
        result.push({ src: srcId, tgt: block.id });
      }
      // Update this column's prev even for branch-starts (so later blocks auto-connect)
      if (!block.same_row_as) colPrev.set(col, block.id);
    } else if (block.same_row_as) {
      // Parallel sibling — inherits the same incoming edge source as same_row_as block
      const sameRowBlock = blocks.find((b) => b.id === block.same_row_as);
      if (sameRowBlock) {
        const sameRowSrcs = sameRowBlock.merge_from;
        if (sameRowSrcs != null) {
          for (const srcId of sameRowSrcs) {
            result.push({ src: srcId, tgt: block.id });
          }
        } else {
          const sameRowCol = sameRowBlock.layout_column ?? 0;
          const prev = colPrev.get(sameRowCol);
          if (prev) result.push({ src: prev, tgt: block.id });
        }
      }
      // Don't update colPrev for sidecars
    } else {
      // Sequential auto-connect from previous block in this column
      const prev = colPrev.get(col);
      if (prev) result.push({ src: prev, tgt: block.id });
      colPrev.set(col, block.id);
    }
  }

  return result;
}

export function buildGraphElements(
  ir: ArchitectureIR,
  expandedBlockIds: Set<string>,
  diff: DiffEntry[]
): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = [];
  const edges: Edge[] = [];
  const diffedIds = getDiffedBlockIds(diff);

  // ── 1. Build Dagre graph ────────────────────────────────────────────────
  const g = new dagre.graphlib.Graph();
  g.setGraph({ rankdir: "TB", ranksep: RANK_SEP, nodesep: NODE_SEP });
  g.setDefaultEdgeLabel(() => ({}));

  for (const block of ir.blocks) {
    g.setNode(block.id, { width: NODE_W, height: NODE_H });
  }

  const edgeList = deriveEdges(ir.blocks);
  for (const { src, tgt } of edgeList) {
    g.setEdge(src, tgt);
  }

  dagre.layout(g);

  // ── 2. Build React Flow nodes from Dagre positions ──────────────────────
  // Dagre returns center-based coordinates; React Flow wants top-left.
  const dagrePositions = new Map<string, { x: number; y: number }>();

  for (const block of ir.blocks) {
    const pos = g.node(block.id);
    if (!pos) continue;
    const x = pos.x - NODE_W / 2;
    const y = pos.y - NODE_H / 2;
    dagrePositions.set(block.id, { x, y });

    const isExpanded = expandedBlockIds.has(block.id);
    const isDiffed   = diffedIds.has(block.id);
    const nodeData: BlockNodeData = {
      block,
      isExpanded,
      isDiffed,
      diffDelta: diffedIds.get(block.id),
    };
    nodes.push({
      id: block.id,
      type: "blockNode",
      position: { x, y },
      data: { data: nodeData },
      style: { width: NODE_W },
    });
  }

  // ── 3. React Flow edges from the derived edge list ──────────────────────
  for (const { src, tgt } of edgeList) {
    edges.push({
      id: `${src}->${tgt}`,
      source: src,
      target: tgt,
      type: "smoothstep",
      style: { stroke: "#475569", strokeWidth: 1.5 },
    });
  }

  // ── 4. Expanded children (placed manually, right of parent) ────────────
  for (const block of ir.blocks) {
    const isExpanded = expandedBlockIds.has(block.id);
    if (!isExpanded || block.children.length === 0) continue;

    const parentPos = dagrePositions.get(block.id);
    if (!parentPos) continue;

    const childX = parentPos.x + NODE_W + CHILD_X_OFFSET;
    let cy = parentPos.y;
    let childPrevId: string | null = null;

    // Residual skip-connection tracking
    const rawLayout   = block.params?.residual_layout as string | undefined;
    const residualLayout =
      rawLayout ||
      (block.children[0]?.type === "layer_norm" ? "pre_ln" : "post_ln");
    const anchorAfterNorm = residualLayout === "post_ln";
    let residualAnchor = block.id;
    let pendingSkipSrc = block.id;

    for (const child of block.children) {
      const childIsExpanded = expandedBlockIds.has(child.id);
      const childNodeId     = `${block.id}__${child.id}`;

      const childNodeData: BlockNodeData = {
        block: child,
        isExpanded: childIsExpanded,
        isDiffed: diffedIds.has(child.id),
        diffDelta: diffedIds.get(child.id),
      };
      nodes.push({
        id: childNodeId,
        type: "blockNode",
        position: { x: childX, y: cy },
        data: { data: childNodeData },
        style: { width: CHILD_W },
      });

      // Edge from parent or previous sibling
      edges.push({
        id: childPrevId
          ? `${childPrevId}->${childNodeId}`
          : `${block.id}->${childNodeId}`,
        source: childPrevId ?? block.id,
        target: childNodeId,
        type: "smoothstep",
        style: {
          stroke: "#6366f1",
          strokeWidth: 1,
          ...(childPrevId ? {} : { strokeDasharray: "4" }),
        },
      });

      // Residual skip edges
      const ct = child.type;
      if (ct === "multi_head_attention" || ct === "feed_forward" || ct === "moe_feed_forward") {
        pendingSkipSrc = residualAnchor;
      } else if (ct === "add") {
        const isParentSrc       = pendingSkipSrc === block.id;
        const isFromPriorAdd    = !isParentSrc && pendingSkipSrc.includes("__add_");
        const useBottomSource   =
          isFromPriorAdd &&
          (residualLayout === "pre_ln" || residualLayout === "t5_decoder");
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
          label: isParentSrc ? "x" : "y",
          data: { residualFromPriorAddBottom: useBottomSource },
        });
        residualAnchor = childNodeId;
      } else if (ct === "layer_norm" && anchorAfterNorm) {
        residualAnchor = childNodeId;
      }

      childPrevId = childNodeId;
      cy += CHILD_H + CHILD_VERT_GAP;

      // Grandchild expansion
      if (childIsExpanded && child.children.length > 0) {
        const gcX = childX + CHILD_W + CHILD_X_OFFSET;
        let gcPrevId: string | null = null;
        const gcParentId = childNodeId;

        for (const gc of child.children) {
          const gcId = `${block.id}__${child.id}__${gc.id}`;
          nodes.push({
            id: gcId,
            type: "blockNode",
            position: { x: gcX, y: cy },
            data: {
              data: {
                block: gc,
                isExpanded: false,
                isDiffed: diffedIds.has(gc.id),
                diffDelta: diffedIds.get(gc.id),
              },
            },
            style: { width: CHILD_W },
          });
          edges.push({
            id: gcPrevId ? `${gcPrevId}->${gcId}` : `${gcParentId}->${gcId}`,
            source: gcPrevId ?? gcParentId,
            target: gcId,
            type: "smoothstep",
            style: {
              stroke: "#a855f7",
              strokeWidth: 1,
              ...(gcPrevId ? {} : { strokeDasharray: "4" }),
            },
          });
          gcPrevId = gcId;
          cy += CHILD_H + CHILD_VERT_GAP;
        }
        cy += CHILD_RANK_SEP;
      }
    }

    // Repeat-count label
    if (block.repeat > 1) {
      const BRACKET_PAD = 8;
      const childrenHeight = cy - CHILD_VERT_GAP - parentPos.y;
      const labelData: RepeatLabelData = {
        repeat: block.repeat,
        heightPx: childrenHeight + BRACKET_PAD * 2,
      };
      nodes.push({
        id: `${block.id}__repeat_label`,
        type: "repeatLabel",
        position: { x: childX + CHILD_W + 16, y: parentPos.y - BRACKET_PAD },
        data: labelData,
        draggable: false,
        selectable: false,
        focusable: false,
      });
    }
  }

  return { nodes, edges };
}
