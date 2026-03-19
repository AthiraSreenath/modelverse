"use client";

"use client";

"use client";

import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  BackgroundVariant,
  type NodeTypes,
  type EdgeTypes,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useEffect, useMemo } from "react";
import { useStore } from "@/lib/store";
import { buildGraphElements } from "./GraphLayout";
import BlockNode from "./nodes/BlockNode";
import RepeatLabelNode from "./nodes/RepeatLabelNode";
import ResidualEdge from "./edges/ResidualEdge";

const NODE_TYPES: NodeTypes = {
  blockNode: BlockNode as NodeTypes[string],
  repeatLabel: RepeatLabelNode as NodeTypes[string],
};

const EDGE_TYPES: EdgeTypes = {
  residual: ResidualEdge as EdgeTypes[string],
};

export default function ArchGraph() {
  const { ir, expandedBlockIds, latestDiff } = useStore();

  const { nodes: nextNodes, edges: nextEdges } = useMemo(() => {
    if (!ir) return { nodes: [], edges: [] };
    return buildGraphElements(ir, expandedBlockIds, latestDiff);
  }, [ir, expandedBlockIds, latestDiff]);

  const [nodes, setNodes, onNodesChange] = useNodesState(nextNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(nextEdges);

  useEffect(() => {
    setNodes(nextNodes);
    setEdges(nextEdges);
  }, [nextNodes, nextEdges, setNodes, setEdges]);

  if (!ir) {
    return (
      <div className="flex-1 flex items-center justify-center text-slate-500 text-sm">
        Enter a model name above to see the architecture
      </div>
    );
  }

  return (
    <div className="flex-1">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={NODE_TYPES}
        edgeTypes={EDGE_TYPES}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        minZoom={0.2}
        maxZoom={2}
        proOptions={{ hideAttribution: true }}
        colorMode="dark"
      >
        <Background
          variant={BackgroundVariant.Dots}
          gap={24}
          size={1}
          color="#334155"
        />
        <Controls
          showInteractive={false}
          className="!bg-slate-800 !border-slate-700 !text-slate-300"
        />
        <MiniMap
          nodeColor={(n) => {
            const type = (n.data?.data as { block?: { type?: string } })?.block?.type;
            const colors: Record<string, string> = {
              embedding: "#3b82f6",
              transformer_stack: "#8b5cf6",
              multi_head_attention: "#a78bfa",
              feed_forward: "#f97316",
              layer_norm: "#94a3b8",
              linear: "#10b981",
            };
            return colors[type ?? ""] ?? "#64748b";
          }}
          maskColor="rgba(15,23,42,0.8)"
          className="!bg-slate-900 !border-slate-700"
        />
      </ReactFlow>
    </div>
  );
}
