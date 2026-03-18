"use client";

import { useMemo } from "react";
import { MessageSquare, Info } from "lucide-react";
import { useStore } from "@/lib/store";
import { getBlockById } from "@/lib/ir";
import { getParamFormulas } from "@/lib/formulas";
import { formatParams } from "@/lib/utils";
import { cn } from "@/lib/utils";

const TYPE_DESCRIPTIONS: Record<string, string> = {
  embedding:
    "Maps discrete token IDs to dense vectors. The vocabulary embedding is the first layer in every transformer — it determines the initial representation of each token.",
  multi_head_attention:
    "Allows the model to attend to different parts of the sequence simultaneously. Multiple attention heads let the model capture different types of relationships in parallel.",
  feed_forward:
    "A two-layer MLP applied independently at each position. After attention aggregates information across tokens, the FFN transforms each position's representation in the same way.",
  moe_feed_forward:
    "Mixture-of-Experts: instead of one FFN, there are many expert networks and a router that selects the top-k most relevant experts per token. This scales parameters without proportionally scaling compute.",
  layer_norm:
    "Normalizes the hidden states within each layer, stabilizing training and enabling deeper networks. RMSNorm (used in LLaMA, Mistral) omits the bias for efficiency.",
  transformer_stack:
    "A repeated block of attention + FFN + residual connections + normalization. The depth (number of repetitions) is the primary determinant of the model's capacity to learn complex representations.",
  linear: "A learned linear projection (matrix multiplication + optional bias).",
  ssm:
    "Structured State Space Model: computes a sequence-to-sequence transformation through a linear recurrence, enabling sub-quadratic sequence modeling (used in Mamba).",
  conv1d:
    "1D depthwise convolution over the sequence dimension. Used in Mamba's selective scan to mix local context efficiently.",
  unknown:
    "Layer type not yet identified. Hover over parameters for raw config values.",
};

export default function DetailPanel() {
  const { ir, selectedBlockId, setSelectedBlockId, isChatStreaming } =
    useStore();

  const block = useMemo(() => {
    if (!ir || !selectedBlockId) return null;
    return getBlockById(ir.blocks, selectedBlockId);
  }, [ir, selectedBlockId]);

  if (!ir) {
    return (
      <div className="h-full flex items-center justify-center p-4">
        <p className="text-sm text-slate-500 text-center">
          Load a model to explore its layers
        </p>
      </div>
    );
  }

  if (!block) {
    return (
      <div className="h-full flex flex-col p-4 gap-3">
        <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider">
          Architecture
        </h2>
        <div className="text-xs text-slate-500 space-y-1">
          <p>
            <span className="text-slate-300 font-medium">{ir.blocks.length}</span>{" "}
            top-level blocks
          </p>
          <p className="text-[11px] text-slate-500">
            Click any node in the graph to inspect it
          </p>
        </div>
        {ir.source_confidence !== "exact" && (
          <div className="mt-2 text-[11px] text-amber-400/80 flex gap-1.5">
            <Info className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
            <span>
              Confidence: <strong>{ir.source_confidence}</strong>. Some details
              may be inferred.
            </span>
          </div>
        )}
      </div>
    );
  }

  const formulas = getParamFormulas(block.type, block.params as Record<string, unknown>);
  const description = TYPE_DESCRIPTIONS[block.type] ?? TYPE_DESCRIPTIONS.unknown;

  return (
    <div className="h-full flex flex-col overflow-y-auto">
      {/* Header */}
      <div className="px-4 pt-4 pb-3 border-b border-slate-800 flex-shrink-0">
        <div className="flex items-start justify-between gap-2">
          <div>
            <h2 className="text-base font-semibold text-white">
              {block.label}
              {block.repeat > 1 && (
                <span className="ml-2 text-sm text-slate-400 font-normal">
                  ×{block.repeat}
                </span>
              )}
            </h2>
            <p className="text-[11px] text-slate-500 font-mono mt-0.5">
              {block.type}
            </p>
          </div>
          {block.param_count != null && (
            <span className="text-sm font-mono font-bold text-white bg-slate-800 px-2 py-1 rounded flex-shrink-0">
              {formatParams(block.param_count * block.repeat)}
            </span>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Description */}
        <p className="text-xs text-slate-400 leading-relaxed">{description}</p>

        {/* Parameter formulas */}
        {formulas.length > 0 && (
          <div>
            <h3 className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-2">
              Parameter breakdown
            </h3>
            <div className="space-y-1.5">
              {formulas.map((f) => (
                <div
                  key={f.label}
                  className="flex justify-between items-baseline gap-2 group"
                >
                  <span className="text-xs text-slate-400 truncate">{f.label}</span>
                  <div className="flex items-baseline gap-1.5 flex-shrink-0">
                    <span className="text-[10px] text-slate-600 font-mono group-hover:text-slate-400 transition-colors">
                      {f.formula}
                    </span>
                    <span className="text-xs font-mono font-semibold text-white">
                      {formatParams(f.value)}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Raw params */}
        {Object.keys(block.params as object).length > 0 && (
          <div>
            <h3 className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-2">
              Configuration
            </h3>
            <div className="space-y-1">
              {Object.entries(block.params as Record<string, unknown>).map(
                ([k, v]) => (
                  <div key={k} className="flex justify-between gap-2">
                    <span className="text-[11px] font-mono text-slate-500 truncate">
                      {k}
                    </span>
                    <span className="text-[11px] font-mono text-slate-300 flex-shrink-0">
                      {String(v)}
                    </span>
                  </div>
                )
              )}
            </div>
          </div>
        )}

        {/* Notes */}
        {block.notes && (
          <div className="text-[11px] text-amber-400/70 bg-amber-400/5 border border-amber-400/20 rounded p-2">
            {block.notes}
          </div>
        )}

        {/* Unknown fields */}
        {block.unknown_fields.length > 0 && (
          <div className="text-[11px] text-slate-500">
            <Info className="w-3 h-3 inline mr-1" />
            Unknown config keys: {block.unknown_fields.join(", ")}
          </div>
        )}
      </div>
    </div>
  );
}
